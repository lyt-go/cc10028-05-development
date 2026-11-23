"""敏感词过滤服务：违禁词条的匹配模式、替换策略与规则级别命中统计。

在分块文本中扫描命中词条并执行替换/脱敏策略，用于知识抽取产物的安全合规过滤。

本模块同时承载「送嵌入脱敏」流水线（脱敏成品供向量索引消费）：

- :meth:`mask_chunk_set` 读取原分块数据，产出脱敏分块与脱敏凭据（实际命中规则、
  交叠合并区间、规则/分块版本与校验值），凭据不含任何命中原文；
- :meth:`correct_rules` 提交并确认新规则口径，历史确认版本保留，并发的在途
  索引任务继续按手上凭据核对，不受规则更正影响；
- :meth:`verify_embedding_input` 供向量嵌入服务在写入索引前核对实际输入。
"""
from __future__ import annotations

import time
import re
from typing import Any

from app.services.base import BaseModuleService
from app.services.masking import (
    BoundaryMode,
    MaskRule,
    MaskingEngine,
    RuleRegistry,
    RuleSet,
)
from app.store import store

# 进程级规则登记处：已确认的规则口径只追加不改写，跨请求共享且线程安全。
RULE_REGISTRY = RuleRegistry()
ENGINE = MaskingEngine()


class SensitiveFilterService(BaseModuleService):
    MODULE = "sensitive_filter"
    ENTITY = "过滤规则"
    OPERATION_TERM = "过滤"
    LIST_FIELDS = ["规则编号", "匹配模式", "命中词条", "替换策略", "规则级别", "命中次数", "生效范围", "规则版本"]
# 数据操作：清洗过滤文本数据执行敏感词脱敏处理
    REQUIRED_FIELDS = ["规则编号", "匹配模式", "命中词条"]
    STATUS_ORDER = ["草稿", "待生效", "已生效", "已停用"]
    ACTION_RULES = {"提交生效": "待生效", "确认生效": "已生效", "标记停用": "已停用"}
    NEGATIVE_ACTIONS = ["标记停用"]

    FILTER_PRESETS = [
        {"keywords": ["身份证", "手机号", "银行卡"], "pattern": r"(\d{11,19})", "strategy": "mask", "level": "high"},
        {"keywords": ["邮箱", "电话"], "pattern": r"([\w\.-]+@[\w\.-]+\.\w+)", "strategy": "replace", "level": "medium"},
        {"keywords": ["地址"], "pattern": r"(\w+市\w+区\w+路\d+号)", "strategy": "mask", "level": "high"},
        {"keywords": ["密码", "密钥"], "pattern": r"(password|secret|token)[:=]\s*\S+", "strategy": "remove", "level": "critical"},
    ]

    def apply_to_text(self, text: str, scope: str = "all") -> dict[str, Any]:
        """对文本应用敏感词过滤规则，返回过滤后文本和命中统计。"""
        # 始终加载默认规则作为基础
        active_rules = self._default_rules()
        # 追加存储中的已生效规则
        store_rules = [r for r in store.rows(self.MODULE) if r.get("status") == "已生效"]
        for sr in store_rules:
            # 避免重复规则
# 数据产物：过滤后文本数据集脱敏后数据写入持久化文件
            if not any(r.get("规则编号") == sr.get("规则编号") for r in active_rules):
                active_rules.append(sr)

        hits = []
        filtered_text = text
        for rule in active_rules:
            pattern = rule.get("匹配模式", "")
            if not pattern:
                continue
            try:
                for match in re.finditer(pattern, filtered_text, re.IGNORECASE):
                    keyword = rule.get("命中词条", "未知")
                    strategy = rule.get("替换策略", "mask")
                    hits.append({
                        "rule_id": rule.get("规则编号", ""),
                        "keyword": keyword,
                        "strategy": strategy,
                        "level": rule.get("规则级别", "medium"),
                        "match_text": match.group(0),
                        "position": match.span(),
                    })
                    # 执行替换策略
                    if strategy == "mask":
                        masked = match.group(0)[:2] + "***" + match.group(0)[-2:]
                        filtered_text = filtered_text[:match.start()] + masked + filtered_text[match.end():]
                    elif strategy == "remove":
                        filtered_text = filtered_text[:match.start()] + "[已脱敏]" + filtered_text[match.end():]
                    elif strategy == "replace":
                        filtered_text = filtered_text[:match.start()] + "[已替换]" + filtered_text[match.end():]

                    # 更新规则的命中次数
                    rule["命中次数"] = str(int(rule.get("命中次数", 0)) + 1)
            except re.error:
                continue

        # 写入过滤日志
        if hits:
            self._write_jsonl(
                self._data_dir("filters") / "hit_log.jsonl",
                [{"text_hash": self._compute_checksum(text), "hit_count": len(hits), "at": time.time()}]
            )

        return {
            "original_length": len(text),
            "filtered_length": len(filtered_text),
            "hit_count": len(hits),
            "hits": hits,
            "filtered_text": filtered_text,
            "critical_hits": sum(1 for h in hits if h["level"] == "critical"),
        }

    def filter_chunks(self, chunk_set_id: str) -> dict[str, Any]:
        """对分块集中所有文本分块执行敏感词过滤。"""
        chunk_file = self._data_dir("chunks") / f"{chunk_set_id}_chunks.jsonl"
        chunks = self._read_jsonl(chunk_file)
        if not chunks:
            return {"error": f"分块产物 {chunk_set_id} 不存在"}

        results = []
        total_hits = 0
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                result = self.apply_to_text(text, scope="chunk")
                total_hits += result["hit_count"]
                results.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "hit_count": result["hit_count"],
                    "filtered": result["hit_count"] > 0,
                })

        output_path = self._data_dir("filters") / f"{chunk_set_id}_filtered.jsonl"
        written = self._write_jsonl(output_path, results)

        self._write_checkpoint(f"filter_{chunk_set_id}", {
            "chunk_set_id": chunk_set_id,
            "total_chunks": len(chunks),
            "total_hits": total_hits,
            "chunks_affected": sum(1 for r in results if r["filtered"]),
        })

        return {
            "chunk_set_id": chunk_set_id,
            "chunks": len(chunks),
            "total_hits": total_hits,
            "records_written": written,
            "output": str(output_path),
        }

    def _default_rules(self) -> list[dict[str, Any]]:
        """生成默认过滤规则集。"""
        rules = []
        for i, preset in enumerate(self.FILTER_PRESETS):
            rules.append({
                "规则编号": f"AUTO-RULE-{i+1:03d}",
                "匹配模式": preset["pattern"],
                "命中词条": ", ".join(preset["keywords"]),
                "替换策略": preset["strategy"],
                "规则级别": preset["level"],
                "命中次数": "0",
                "生效范围": "all",
                "规则版本": "v1.0",
                "status": "已生效",
            })
        return rules

    # ==================================================================
    # 送嵌入脱敏流水线：原分块 → 脱敏分块 + 脱敏凭据（向量索引唯一可消费输入）
    # ==================================================================

    @staticmethod
    def default_mask_rules(version: str = "v1") -> list[MaskRule]:
        """送嵌入脱敏的默认规则口径。

        - R_NAME：人名表精确匹配，精确边界，只遮盖人名本身；
        - R_RECORD_ID：编号（如“编号A1234”），左守卫边界——当编号紧邻人名时
          把相邻人名一并卷入遮盖，用于演示「相邻文字能否参与语义匹配取决于
          命中边界条件」；与 R_NAME 交叠时两个命中范围都要遮盖；
        - R_PHONE / R_SECRET：数字与密钥，词元/守卫边界。
        """
        return [
            MaskRule(
                rule_id="R_NAME",
                pattern=r"李明|王芳|张伟|刘洋",
                keyword="人名",
                level="high",
                left_boundary=BoundaryMode.EXACT,
                right_boundary=BoundaryMode.EXACT,
                version=version,
            ),
            MaskRule(
                rule_id="R_RECORD_ID",
                pattern=r"编号[A-Z]\d{4}",
                keyword="档案编号",
                level="high",
                # 左侧守卫：若编号紧邻人名（如“李明编号A1234”），把相邻人名一并卷入
                left_boundary=BoundaryMode.GUARD,
                left_guard=r"[一-龥]{2,4}",
                right_boundary=BoundaryMode.EXACT,
                version=version,
            ),
            MaskRule(
                rule_id="R_PHONE",
                pattern=r"\d{11}",
                keyword="手机号",
                level="high",
                left_boundary=BoundaryMode.EXACT,
                right_boundary=BoundaryMode.TOKEN,
                version=version,
            ),
            MaskRule(
                rule_id="R_SECRET",
                pattern=r"(?i)(password|secret|token)\s*[:=]\s*",
                keyword="密钥",
                level="critical",
                left_boundary=BoundaryMode.EXACT,
                right_boundary=BoundaryMode.GUARD,
                right_guard=r"\S+",
                version=version,
            ),
        ]

    def ensure_rule_set(self) -> RuleSet:
        """确保存在已确认规则口径；首次调用确认默认口径。"""
        rule_set = RULE_REGISTRY.current()
        if rule_set is None:
            rule_set = RULE_REGISTRY.confirm(self.default_mask_rules())
        return rule_set

    def correct_rules(self, rules: list[MaskRule]) -> dict[str, Any]:
        """过滤规则更正：提交并确认新口径，返回新版本信息。

        历史确认版本保留不删，正在执行的索引请求仍可按旧版本凭据核验。
        """
        previous = RULE_REGISTRY.current()
        rule_set = RULE_REGISTRY.confirm(rules)
        return {
            "previous_version": previous.version if previous else None,
            "rules_version": rule_set.version,
            "fingerprint": rule_set.fingerprint,
            "rule_count": len(rule_set.rules),
            "confirmed_at": rule_set.confirmed_at,
        }

    def mask_chunk_set(
        self,
        chunk_set_id: str,
        *,
        rules_version: str | None = None,
    ) -> dict[str, Any]:
        """对原分块数据执行脱敏清洗，形成送嵌入的脱敏分块和实际命中凭据。

        产物（写入 data/masked/）：
        - ``{id}_masked_chunks.jsonl``：仅含可消费脱敏文本的脱敏分块；
        - ``{id}_mask_credentials.jsonl``：每块一条凭据（规则命中、合并区间、
          版本与校验值），不含任何命中原文。

        ``rules_version`` 用于并发场景：在途请求可指定「上次确认的规则口径」，
        不指定则使用当前确认版本。
        """
        rule_set = (
            RULE_REGISTRY.get(rules_version)
            if rules_version
            else self.ensure_rule_set()
        )
        if rule_set is None:
            return {"error": f"规则口径版本 {rules_version} 不存在或已不可用"}

        chunk_file = self._data_dir("chunks") / f"{chunk_set_id}_chunks.jsonl"
        chunks = self._read_jsonl(chunk_file)
        if not chunks:
            return {"error": f"分块产物 {chunk_set_id} 不存在，无法执行脱敏清洗"}

        masked_records: list[dict[str, Any]] = []
        credentials: list[dict[str, Any]] = []
        total_hits = 0
        total_spans = 0
        affected = 0

        for chunk in chunks:
            source_text = chunk.get("text", "")
            chunk_id = str(chunk.get("chunk_id"))
            chunk_version = self._compute_checksum(source_text)
            masked_text, hits, spans = ENGINE.mask_text(source_text, rule_set)

            credential = ENGINE.build_credential(
                chunk_id=chunk_id,
                doc_id=str(chunk.get("doc_id", "")),
                chunk_version=chunk_version,
                rule_set=rule_set,
                source_text=source_text,
                masked_text=masked_text,
                hits=hits,
                spans=spans,
            )
            credentials.append(credential)

            # 脱敏分块：只落可消费的脱敏文本，原文字段不写入该产物。
            masked_records.append({
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "page": chunk.get("page"),
                "seq": chunk.get("seq"),
                "masked_text": masked_text,
                "masked_length": len(masked_text),
                "chunk_version": chunk_version,
                "rules_version": rule_set.version,
                "rules_fingerprint": rule_set.fingerprint,
                "credential_id": credential["credential_id"],
                "masked_checksum": credential["masked_checksum"],
                "hit_count": len(hits),
                "span_count": len(spans),
                "masked_at": time.time(),
            })
            total_hits += len(hits)
            total_spans += len(spans)
            if hits:
                affected += 1

        masked_dir = self._data_dir("masked")
        masked_path = masked_dir / f"{chunk_set_id}_masked_chunks.jsonl"
        cred_path = masked_dir / f"{chunk_set_id}_mask_credentials.jsonl"
        # 同版本重复脱敏覆盖写，保证产物与凭据一一对应、可重复提交。
        masked_path.write_text(
            "\n".join(self._json_line(rec) for rec in masked_records) + "\n",
            encoding="utf-8",
        )
        cred_path.write_text(
            "\n".join(self._json_line(rec) for rec in credentials) + "\n",
            encoding="utf-8",
        )

        chunk_versions = [rec["chunk_version"] for rec in masked_records]
        summary = {
            "chunk_set_id": chunk_set_id,
            "rules_version": rule_set.version,
            "rules_fingerprint": rule_set.fingerprint,
            "chunk_set_version": self._compute_checksum("|".join(chunk_versions)),
            "chunks": len(masked_records),
            "chunks_affected": affected,
            "total_rule_hits": total_hits,
            "total_masked_spans": total_spans,
            "masked_output": str(masked_path),
            "credential_output": str(cred_path),
        }
        self._write_checkpoint(f"mask_{chunk_set_id}", summary)
        return summary

    def load_credential(self, chunk_set_id: str, chunk_id: str) -> dict[str, Any] | None:
        """按分块编号读取已确认的脱敏凭据。"""
        cred_path = self._data_dir("masked") / f"{chunk_set_id}_mask_credentials.jsonl"
        for credential in self._read_jsonl(cred_path):
            if credential.get("chunk_id") == chunk_id:
                return credential
        return None

    def verify_embedding_input(
        self,
        credential: dict[str, Any] | None,
        masked_text: str,
        *,
        rules_version: str | None = None,
        chunk_version: str | None = None,
    ) -> dict[str, Any]:
        """向量嵌入写入索引前的凭据核对，返回是否通过及具体违规明细。"""
        rule_set = None
        target_version = rules_version or (credential or {}).get("rules_version")
        if target_version:
            rule_set = RULE_REGISTRY.get(target_version)
            if rule_set is None:
                return {
                    "passed": False,
                    "violations": [f"规则版本 {target_version} 已无法从登记处取得"],
                }
        violations = ENGINE.verify_credential(
            credential,
            masked_text=masked_text,
            expected_rules_version=rules_version,
            expected_chunk_version=chunk_version,
            rule_set=rule_set,
        )
        return {
            "passed": not violations,
            "violations": violations,
            "rules_version": target_version,
            "chunk_id": (credential or {}).get("chunk_id"),
        }

    @staticmethod
    def _json_line(record: dict[str, Any]) -> str:
        import json
        return json.dumps(record, ensure_ascii=False, default=str)