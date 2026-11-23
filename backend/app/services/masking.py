"""脱敏引擎：规则口径版本化、多规则交叠命中合并、脱敏凭据生成与核验。

供敏感词过滤服务产出「送嵌入的脱敏分块 + 脱敏凭据」，供向量嵌入服务在写入
索引前核对实际输入。设计上落实以下安全约束：

1. 实际命中的字符绝不进入向量输入。所有命中范围（含多条规则交叠、相邻的
   范围）合并后统一用定长占位符 ``MASK_TOKEN`` 替换，脱敏文本不残留任何被
   命中的原文字符，也不泄露被遮盖片段的长度信息。
2. 脱敏凭据只记录规则编号、相对原始分块的偏移区间与边界口径，不保存任何
   命中原文；重叠原文无法通过片段拼接或凭据反推还原。
3. 每条规则可独立配置左右边界（exact / token / guard），决定命中位置旁边
   的相邻文字是否参与（被卷入）遮盖范围。
4. 规则集带不可变版本与口径指纹。规则更正与索引请求并发时，在途任务持有
   的已确认规则集快照不受影响，历史确认版本保留以便旧凭据继续核验。
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

# 定长脱敏占位符：不含任何原文字符，所有命中范围统一替换为它，
# 使向量输入侧无法从占位符推断被遮盖内容或其长度。
MASK_TOKEN = "〔MASK〕"
# 自定义守卫正则向相邻文字吸收字符时的上限，防止失控吞掉整段文本。
GUARD_ABSORB_CAP = 64
# 词元边界只吸收 ASCII 字母数字下划线；中文等字符不算词元，避免整句被卷入。
_TOKEN_CHAR = re.compile(r"[A-Za-z0-9_]")


class BoundaryMode:
    """命中边界口径：决定相邻文字能否参与语义匹配/是否被卷入遮盖。"""

    EXACT = "exact"    # 精确边界：仅遮盖正则本身命中的字符，相邻文字保留
    TOKEN = "token"    # 词元边界：向两侧吸收相邻的连续词字符（含中文/字母数字）
    GUARD = "guard"    # 守卫边界：按规则自带的守卫正则吸收相邻字符


@dataclass(frozen=True)
class MaskRule:
    """一条脱敏规则的完整口径。frozen 保证规则一旦确认即不可变。"""

    rule_id: str
    pattern: str
    keyword: str = ""
    level: str = "medium"
    left_boundary: str = BoundaryMode.EXACT
    right_boundary: str = BoundaryMode.EXACT
    left_guard: str | None = None
    right_guard: str | None = None
    ignorecase: bool = False
    version: str = "v1"

    def compiled(self) -> re.Pattern[str] | None:
        """编译规则正则；非法口径返回 None，由引擎跳过而不是中断整条链路。"""
        try:
            flags = re.IGNORECASE if self.ignorecase else 0
            return re.compile(self.pattern, flags)
        except re.error:
            return None

    def canonical(self) -> dict[str, Any]:
        """规范化口径字典，参与规则集指纹计算。"""
        return {
            "rule_id": self.rule_id,
            "pattern": self.pattern,
            "keyword": self.keyword,
            "level": self.level,
            "left_boundary": self.left_boundary,
            "right_boundary": self.right_boundary,
            "left_guard": self.left_guard,
            "right_guard": self.right_guard,
            "ignorecase": self.ignorecase,
            "version": self.version,
        }


@dataclass(frozen=True)
class RuleSet:
    """某一版已确认的规则口径。整体不可变，可被在途任务长期持有。"""

    version: str
    rules: tuple[MaskRule, ...]
    fingerprint: str
    confirmed_at: float = 0.0

    def get(self, rule_id: str) -> MaskRule | None:
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None


@dataclass(frozen=True)
class HitRecord:
    """单条规则的一次实际命中（偏移相对原始分块，半开区间）。"""

    rule_id: str
    start: int
    end: int
    level: str
    left_boundary: str
    right_boundary: str
    expanded: bool          # 边界口径是否向相邻文字扩展过
    keyword: str = ""


@dataclass(frozen=True)
class MaskedSpan:
    """合并后的遮盖区间：交叠/相邻的多个命中合并为一个不可分割的遮盖段。"""

    start: int
    end: int
    rule_ids: tuple[str, ...]
    levels: tuple[str, ...]


class RuleRegistry:
    """规则口径登记处：草稿提交、确认生效、历史版本保留。

    线程安全。``confirm`` 只会新增不可变版本，从不改写已确认版本，因此
    规则更正与索引请求并发时，上次确认的口径仍可被在途任务继续使用。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._confirmed: dict[str, RuleSet] = {}
        self._current_version: str | None = None
        self._seq = 0

    @staticmethod
    def fingerprint_of(rules: Iterable[MaskRule]) -> str:
        payload = json.dumps(
            [r.canonical() for r in sorted(rules, key=lambda r: r.rule_id)],
            ensure_ascii=False, sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def submit(self, rules: Iterable[MaskRule]) -> RuleSet:
        """提交一版候选口径并做合法性校验，尚未生效，不影响当前确认版本。"""
        rule_tuple = tuple(rules)
        for rule in rule_tuple:
            if rule.compiled() is None:
                raise ValueError(f"规则 {rule.rule_id} 的匹配模式非法：{rule.pattern}")
        return RuleSet(
            version="candidate",
            rules=rule_tuple,
            fingerprint=self.fingerprint_of(rule_tuple),
        )

    def confirm(self, rules: Iterable[MaskRule] | None = None) -> RuleSet:
        """确认规则生效。口径与当前版本一致时幂等返回当前版本，否则生成新版本。"""
        with self._lock:
            if rules is None:
                if self._current_version is None:
                    raise RuntimeError("尚无已确认的规则口径")
                return self._confirmed[self._current_version]
            candidate = self.submit(rules)
            current = self.current()
            if current is not None and candidate.fingerprint == current.fingerprint:
                return current
            self._seq += 1
            version = f"rv-{self._seq:04d}"
            rule_set = RuleSet(
                version=version,
                rules=candidate.rules,
                fingerprint=candidate.fingerprint,
                confirmed_at=time.time(),
            )
            # 只追加不改写：历史确认版本永久保留，供旧脱敏凭据核验。
            self._confirmed[version] = rule_set
            self._current_version = version
            return rule_set

    def current(self) -> RuleSet | None:
        with self._lock:
            if self._current_version is None:
                return None
            return self._confirmed[self._current_version]

    def get(self, version: str) -> RuleSet | None:
        with self._lock:
            return self._confirmed.get(version)


class MaskingEngine:
    """无状态脱敏引擎：扫描命中、按边界口径扩展、合并交叠区间、生成/核验凭据。"""

    MASK_TOKEN = MASK_TOKEN

    # -- 脱敏主流程 --

    def mask_text(
        self, text: str, rule_set: RuleSet
    ) -> tuple[str, list[HitRecord], list[MaskedSpan]]:
        """对原始分块执行脱敏，返回（脱敏文本，规则命中明细，合并遮盖区间）。"""
        raw_hits: list[tuple[int, int, MaskRule, bool]] = []
        for rule in rule_set.rules:
            pattern = rule.compiled()
            if pattern is None:
                continue
            for match in pattern.finditer(text):
                start, end = match.span()
                new_start, new_end, expanded = self._expand_span(text, start, end, rule)
                raw_hits.append((new_start, new_end, rule, expanded))

        hits = [
            HitRecord(
                rule_id=rule.rule_id,
                start=start,
                end=end,
                level=rule.level,
                left_boundary=rule.left_boundary,
                right_boundary=rule.right_boundary,
                expanded=expanded,
                keyword=rule.keyword,
            )
            for start, end, rule, expanded in raw_hits
        ]
        spans = self._merge_spans(raw_hits)
        masked_text = self._apply_spans(text, spans)
        return masked_text, hits, spans

    def _expand_span(
        self, text: str, start: int, end: int, rule: MaskRule
    ) -> tuple[int, int, bool]:
        """按规则左右边界口径决定相邻文字是否被吸收进命中范围。"""
        original = (start, end)

        if rule.left_boundary == BoundaryMode.TOKEN:
            while start > 0 and _TOKEN_CHAR.match(text[start - 1]):
                start -= 1
        elif rule.left_boundary == BoundaryMode.GUARD and rule.left_guard:
            try:
                match = re.search(r"(?:" + rule.left_guard + r")\Z", text[:start])
            except re.error:
                match = None
            if match and len(match.group(0)) <= GUARD_ABSORB_CAP:
                start = match.start()

        if rule.right_boundary == BoundaryMode.TOKEN:
            while end < len(text) and _TOKEN_CHAR.match(text[end]):
                end += 1
        elif rule.right_boundary == BoundaryMode.GUARD and rule.right_guard:
            try:
                match = re.match(rule.right_guard, text[end:])
            except re.error:
                match = None
            if match and len(match.group(0)) <= GUARD_ABSORB_CAP:
                end += len(match.group(0))

        return start, end, (start, end) != original

    @staticmethod
    def _merge_spans(
        raw_hits: list[tuple[int, int, MaskRule, bool]],
    ) -> list[MaskedSpan]:
        """合并真正交叠的命中；首尾相邻但不重叠的命中保留为独立遮盖段。

        两种情况下被命中的原文字符都不会进入脱敏文本；保留相邻段的独立性，
        使两条规则各自的命中边界口径在凭据中可区分、可核对。重叠原文因区间
        合并为同一定长占位符，无法通过片段拼接还原。
        """
        ordered = sorted(raw_hits, key=lambda item: (item[0], item[1]))
        merged: list[dict[str, Any]] = []
        for start, end, rule, _ in ordered:
            if merged and start < merged[-1]["end"]:
                prev = merged[-1]
                prev["end"] = max(prev["end"], end)
                if rule.rule_id not in prev["rule_ids"]:
                    prev["rule_ids"].append(rule.rule_id)
                    prev["levels"].append(rule.level)
            else:
                merged.append({
                    "start": start, "end": end,
                    "rule_ids": [rule.rule_id], "levels": [rule.level],
                })
        return [
            MaskedSpan(
                start=item["start"],
                end=item["end"],
                rule_ids=tuple(item["rule_ids"]),
                levels=tuple(item["levels"]),
            )
            for item in merged
        ]

    def _apply_spans(self, text: str, spans: list[MaskedSpan]) -> str:
        """按合并区间自后向前替换为定长占位符，保证命中字符全部离开脱敏文本。"""
        result = text
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            result = result[:span.start] + self.MASK_TOKEN + result[span.end:]
        return result

    # -- 凭据 --

    @staticmethod
    def _checksum(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def build_credential(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        chunk_version: str,
        rule_set: RuleSet,
        source_text: str,
        masked_text: str,
        hits: list[HitRecord],
        spans: list[MaskedSpan],
    ) -> dict[str, Any]:
        """生成脱敏凭据：只含偏移与口径，不含任何命中原文。"""
        credential = {
            "credential_id": f"CRED-{chunk_id}-{rule_set.version}",
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "chunk_version": chunk_version,
            "rules_version": rule_set.version,
            "rules_fingerprint": rule_set.fingerprint,
            "source_length": len(source_text),
            "masked_length": len(masked_text),
            "source_checksum": self._checksum(source_text),
            "masked_checksum": self._checksum(masked_text),
            "mask_token": self.MASK_TOKEN,
            "hits": [
                {
                    "rule_id": hit.rule_id,
                    "start": hit.start,
                    "end": hit.end,
                    "level": hit.level,
                    "left_boundary": hit.left_boundary,
                    "right_boundary": hit.right_boundary,
                    "expanded": hit.expanded,
                    "keyword": hit.keyword,
                }
                for hit in hits
            ],
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "rule_ids": list(span.rule_ids),
                    "levels": list(span.levels),
                }
                for span in spans
            ],
            "created_at": time.time(),
        }
        credential["fingerprint"] = self._fingerprint(credential)
        return credential

    @staticmethod
    def _fingerprint(credential: dict[str, Any]) -> str:
        payload = {k: v for k, v in credential.items() if k != "fingerprint"}
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify_credential(
        self,
        credential: dict[str, Any] | None,
        *,
        masked_text: str | None = None,
        expected_rules_version: str | None = None,
        expected_chunk_version: str | None = None,
        rule_set: RuleSet | None = None,
    ) -> list[str]:
        """核验脱敏凭据与实际送嵌入输入是否一致，返回违规说明列表（空=通过）。"""
        violations: list[str] = []
        if not credential:
            return ["凭据不存在"]

        recomputed = self._fingerprint(credential)
        if recomputed != credential.get("fingerprint"):
            violations.append("凭据指纹不匹配：凭据内容可能被篡改")

        if expected_rules_version and credential.get("rules_version") != expected_rules_version:
            violations.append(
                f"规则版本不符：凭据 {credential.get('rules_version')} / 请求 {expected_rules_version}"
            )
        if expected_chunk_version and credential.get("chunk_version") != expected_chunk_version:
            violations.append(
                f"分块版本不符：凭据 {credential.get('chunk_version')} / 请求 {expected_chunk_version}"
            )
        if rule_set is not None:
            if credential.get("rules_version") != rule_set.version:
                violations.append(
                    f"凭据规则版本 {credential.get('rules_version')} 不在手上的规则口径 {rule_set.version} 内"
                )
            elif credential.get("rules_fingerprint") != rule_set.fingerprint:
                violations.append("规则口径指纹不匹配：规则内容与确认版本不一致")

        if masked_text is not None:
            if self._checksum(masked_text) != credential.get("masked_checksum"):
                violations.append("脱敏文本校验值不匹配：实际输入与凭据登记的脱敏分块不一致")
            token_count = masked_text.count(self.MASK_TOKEN)
            if token_count != len(credential.get("spans", [])):
                violations.append(
                    f"占位符数量 {token_count} 与遮盖区间数 {len(credential.get('spans', []))} 不一致"
                )
            if len(masked_text) != credential.get("masked_length"):
                violations.append("脱敏文本长度与凭据登记不一致")

        for span in credential.get("spans", []):
            if not (0 <= span["start"] < span["end"] <= credential.get("source_length", 0)):
                violations.append(f"遮盖区间越界：{span['start']}:{span['end']}")
        return violations

    def reconstruct_masked_text(
        self, source_text: str, credential: dict[str, Any]
    ) -> str | None:
        """用原始分块+凭据重建脱敏文本；仅用于脱敏侧自检，向量侧拿不到原文。"""
        if self._checksum(source_text) != credential.get("source_checksum"):
            return None
        result = source_text
        for span in sorted(credential["spans"], key=lambda item: item["start"], reverse=True):
            result = result[:span["start"]] + self.MASK_TOKEN + result[span["end"]:]
        return result
