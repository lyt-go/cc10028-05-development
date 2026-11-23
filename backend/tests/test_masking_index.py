"""送嵌入脱敏链路测试。

覆盖：
- 多规则交叠命中（人名 + 覆盖相邻编号的规则）范围全部遮盖；
- 命中边界条件决定相邻文字是否参与（卷入）遮盖；
- 实际命中原文字符不进入脱敏文本，重叠原文无法由凭据/片段拼接还原；
- 有交叠 / 仅一条规则命中 / 无命中三段文字的过滤凭据与索引输入比对；
- 规则更正与索引请求并发时保留上次确认口径与分块版本；
- 向量嵌入只消费脱敏文本且逐条凭据核对，篡改输入被整批拒绝；
- 重新提交索引请求后新旧模型输入的命中明细比对。
"""
import os
import sys
import threading

import pytest

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from app.services.base import DATA_ROOT
from app.services.masking import BoundaryMode, MaskRule
from app.services.sensitive_filter import ENGINE, RULE_REGISTRY, SensitiveFilterService
from app.services.vector_embed import VectorEmbedService

# 三段固定文字：交叠命中 / 仅一条规则命中 / 无命中
SEG_OVERLAP = "李明编号A1234的档案已入库"
SEG_SINGLE = "联系电话13800138000请尽快拨打"
SEG_CLEAN = "今日天气晴好适合户外团建活动"


def _write_chunks(chunk_set_id: str) -> list[dict]:
    svc = SensitiveFilterService()
    chunks = [
        {"chunk_id": f"{chunk_set_id}-C1", "doc_id": "DOC-MASK", "page": 1,
         "seq": 1, "text": SEG_OVERLAP},
        {"chunk_id": f"{chunk_set_id}-C2", "doc_id": "DOC-MASK", "page": 1,
         "seq": 2, "text": SEG_SINGLE},
        {"chunk_id": f"{chunk_set_id}-C3", "doc_id": "DOC-MASK", "page": 1,
         "seq": 3, "text": SEG_CLEAN},
    ]
    path = svc._data_dir("chunks") / f"{chunk_set_id}_chunks.jsonl"
    path.write_text(
        "\n".join(svc._json_line(rec) for rec in chunks) + "\n", encoding="utf-8"
    )
    return chunks


@pytest.fixture(autouse=True)
def reset_rule_registry():
    """每个用例前重置全局规则登记处，避免规则更正在用例间串状态。"""
    with RULE_REGISTRY._lock:
        RULE_REGISTRY._confirmed.clear()
        RULE_REGISTRY._current_version = None
        RULE_REGISTRY._seq = 0
    yield


@pytest.fixture()
def rule_set_v1():
    svc = SensitiveFilterService()
    return svc.ensure_rule_set()


class TestMaskingEngineUnit:
    """脱敏引擎单元行为。"""

    def test_overlapping_rules_cover_full_range(self, rule_set_v1):
        masked, hits, spans = ENGINE.mask_text(SEG_OVERLAP, rule_set_v1)
        rule_ids = {hit.rule_id for hit in hits}
        # 人名规则与编号规则同时命中
        assert {"R_NAME", "R_RECORD_ID"} <= rule_ids
        # 交叠命中合并为一个遮盖区间，范围覆盖人名与编号
        assert len(spans) == 1
        span = spans[0]
        assert set(span.rule_ids) == {"R_NAME", "R_RECORD_ID"}
        assert SEG_OVERLAP[span.start:span.end] == "李明编号A1234"
        # 实际命中字符全部离开脱敏文本
        assert "李明" not in masked
        assert "编号A1234" not in masked
        assert masked == ENGINE.MASK_TOKEN + "的档案已入库"

    def test_single_rule_hit_segment(self, rule_set_v1):
        masked, hits, spans = ENGINE.mask_text(SEG_SINGLE, rule_set_v1)
        assert {hit.rule_id for hit in hits} == {"R_PHONE"}
        assert len(spans) == 1
        # 精确左边界：相邻文字“电话”保留；右边界词元吸收：数字串整体遮盖
        assert SEG_SINGLE[spans[0].start:spans[0].end] == "13800138000"
        assert "电话" in masked and "请尽快拨打" in masked
        assert "13800138000" not in masked

    def test_no_hit_segment_passthrough(self, rule_set_v1):
        masked, hits, spans = ENGINE.mask_text(SEG_CLEAN, rule_set_v1)
        assert hits == [] and spans == []
        assert masked == SEG_CLEAN
        assert ENGINE.MASK_TOKEN not in masked

    def test_boundary_mode_decides_adjacent_chars(self):
        text = "李明编号A1234"
        exact = MaskRule(
            rule_id="R_ID_EXACT", pattern=r"编号[A-Z]\d{4}",
            left_boundary=BoundaryMode.EXACT, right_boundary=BoundaryMode.EXACT,
        )
        guard = MaskRule(
            rule_id="R_ID_GUARD", pattern=r"编号[A-Z]\d{4}",
            left_boundary=BoundaryMode.GUARD, left_guard=r"[一-龥]{2,4}",
            right_boundary=BoundaryMode.EXACT,
        )
        rs_exact = RULE_REGISTRY.submit([exact])
        rs_guard = RULE_REGISTRY.submit([guard])
        masked_exact, _, spans_exact = ENGINE.mask_text(text, rs_exact)
        masked_guard, _, spans_guard = ENGINE.mask_text(text, rs_guard)
        # 精确边界：相邻人名保留并可参与语义匹配
        assert spans_exact[0].start == 2
        assert masked_exact == "李明" + ENGINE.MASK_TOKEN
        # 守卫边界：相邻人名被卷入遮盖，不能再参与语义匹配
        assert spans_guard[0].start == 0
        assert masked_guard == ENGINE.MASK_TOKEN

    def test_credential_contains_no_hit_text(self, rule_set_v1):
        masked, hits, spans = ENGINE.mask_text(SEG_OVERLAP, rule_set_v1)
        cred = ENGINE.build_credential(
            chunk_id="c", doc_id="d", chunk_version="v", rule_set=rule_set_v1,
            source_text=SEG_OVERLAP, masked_text=masked, hits=hits, spans=spans,
        )
        serialized = SensitiveFilterService._json_line(cred)
        for leaked in ("李明", "编号A1234"):
            assert leaked not in serialized
        # 凭据只保留偏移和规则口径
        assert cred["spans"][0]["start"] == 0
        assert set(cred["spans"][0]["rule_ids"]) == {"R_NAME", "R_RECORD_ID"}

    def test_reconstruction_from_fragments_impossible(self, rule_set_v1):
        masked, _, spans = ENGINE.mask_text(SEG_OVERLAP, rule_set_v1)
        # 脱敏文本（剔除占位符本身）中不存在任何被遮盖字符，无法通过片段拼接还原
        residual = masked.replace(ENGINE.MASK_TOKEN, "")
        covered = set("".join(SEG_OVERLAP[s.start:s.end] for s in spans))
        for char in covered:
            assert char not in residual


class TestMaskingPipeline:
    """脱敏分块与凭据的端到端产物。"""

    def test_mask_chunk_set_artifacts(self, rule_set_v1):
        chunk_set_id = "MASK-E2E"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        summary = svc.mask_chunk_set(chunk_set_id)
        assert "error" not in summary
        assert summary["chunks"] == 3
        assert summary["chunks_affected"] == 2  # 第三段无命中
        assert summary["total_rule_hits"] >= 3  # 人名+编号+手机号

        masked_path = DATA_ROOT / "masked" / f"{chunk_set_id}_masked_chunks.jsonl"
        cred_path = DATA_ROOT / "masked" / f"{chunk_set_id}_mask_credentials.jsonl"
        assert masked_path.exists() and cred_path.exists()

        masked_rows = svc._read_jsonl(masked_path)
        cred_rows = svc._read_jsonl(cred_path)
        assert len(masked_rows) == len(cred_rows) == 3
        by_id = {row["chunk_id"]: row for row in masked_rows}
        # 三段文字的过滤凭据与索引输入比对
        assert ENGINE.MASK_TOKEN in by_id[f"{chunk_set_id}-C1"]["masked_text"]
        assert ENGINE.MASK_TOKEN in by_id[f"{chunk_set_id}-C2"]["masked_text"]
        assert ENGINE.MASK_TOKEN not in by_id[f"{chunk_set_id}-C3"]["masked_text"]
        # 脱敏分块产物不含原文字段
        for row in masked_rows:
            assert "text" not in row
        # 凭据文件不含命中原文
        raw_cred = cred_path.read_text(encoding="utf-8")
        assert "李明" not in raw_cred and "13800138000" not in raw_cred

    def test_self_consistency_rebuild(self, rule_set_v1):
        rule_set = SensitiveFilterService().ensure_rule_set()
        masked, hits, spans = ENGINE.mask_text(SEG_OVERLAP, rule_set)
        cred = ENGINE.build_credential(
            chunk_id="c", doc_id="d", chunk_version="v", rule_set=rule_set,
            source_text=SEG_OVERLAP, masked_text=masked, hits=hits, spans=spans,
        )
        rebuilt = ENGINE.reconstruct_masked_text(SEG_OVERLAP, cred)
        assert rebuilt == masked
        # 原文被调换后重建必须失败
        assert ENGINE.reconstruct_masked_text(SEG_OVERLAP + "x", cred) is None


class TestCredentialVerification:
    """向量写入前的凭据核对。"""

    def test_verify_passes_for_authorized_input(self):
        chunk_set_id = "MASK-VERIFY-OK"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        masked_rows = svc._read_jsonl(
            DATA_ROOT / "masked" / f"{chunk_set_id}_masked_chunks.jsonl"
        )
        for row in masked_rows:
            cred = svc.load_credential(chunk_set_id, row["chunk_id"])
            check = svc.verify_embedding_input(cred, row["masked_text"])
            assert check["passed"], check["violations"]

    def test_tampered_input_rejected(self):
        chunk_set_id = "MASK-VERIFY-TAMPER"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        masked_rows = svc._read_jsonl(
            DATA_ROOT / "masked" / f"{chunk_set_id}_masked_chunks.jsonl"
        )
        target = next(r for r in masked_rows if r["chunk_id"].endswith("-C1"))
        cred = svc.load_credential(chunk_set_id, target["chunk_id"])
        # 把命中原文字符塞回送嵌入输入
        tampered = target["masked_text"].replace(ENGINE.MASK_TOKEN, "李明编号A1234", 1)
        check = svc.verify_embedding_input(cred, tampered)
        assert check["passed"] is False
        assert any("校验值" in v for v in check["violations"])

    def test_tampered_credential_rejected(self):
        chunk_set_id = "MASK-CRED-TAMPER"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        target_id = f"{chunk_set_id}-C1"
        cred = svc.load_credential(chunk_set_id, target_id)
        masked_rows = {
            r["chunk_id"]: r
            for r in svc._read_jsonl(
                DATA_ROOT / "masked" / f"{chunk_set_id}_masked_chunks.jsonl"
            )
        }
        cred["spans"][0]["end"] += 1  # 篡改遮盖区间
        check = svc.verify_embedding_input(cred, masked_rows[target_id]["masked_text"])
        assert check["passed"] is False
        assert any("指纹" in v for v in check["violations"])

    def test_embed_rejects_batch_on_violation(self):
        chunk_set_id = "MASK-EMBED-REJECT"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        # 直接破坏脱敏产物，模拟未授权文本流入
        masked_path = DATA_ROOT / "masked" / f"{chunk_set_id}_masked_chunks.jsonl"
        rows = svc._read_jsonl(masked_path)
        rows[0]["masked_text"] = rows[0]["masked_text"].replace(
            ENGINE.MASK_TOKEN, "李明编号A1234", 1
        )
        masked_path.write_text(
            "\n".join(svc._json_line(r) for r in rows) + "\n", encoding="utf-8"
        )
        result = VectorEmbedService().embed_masked_artifacts(chunk_set_id)
        assert "error" in result
        assert result["written"] == 0
        assert result["rejected"][0]["chunk_id"].endswith("-C1")


class TestConcurrencyAndResubmission:
    """规则更正与索引请求并发、重新提交后的新旧命中明细。"""

    def test_inflight_request_keeps_last_confirmed_rules(self):
        chunk_set_id = "MASK-CONCUR"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        first = svc.mask_chunk_set(chunk_set_id)
        pinned_version = first["rules_version"]

        barrier = threading.Barrier(2)
        holder: dict = {}

        def inflight_index():
            # 在途任务持有上次确认的规则版本与分块版本
            rule_set = RULE_REGISTRY.get(pinned_version)
            barrier.wait()  # 与规则更正同时发生
            holder["rule_set_version"] = rule_set.version
            holder["fingerprint"] = rule_set.fingerprint
            result = VectorEmbedService().embed_masked_artifacts(
                chunk_set_id, rules_version=pinned_version
            )
            holder["result"] = result

        thread = threading.Thread(target=inflight_index)
        thread.start()
        barrier.wait()
        # 并发地更正规则：编号规则从守卫边界改为精确边界
        corrected = [
            MaskRule(rule_id="R_NAME", pattern=r"李明|王芳|张伟|刘洋", keyword="人名",
                     level="high"),
            MaskRule(rule_id="R_RECORD_ID", pattern=r"编号[A-Z]\d{4}",
                     keyword="档案编号", level="high"),
            MaskRule(rule_id="R_PHONE", pattern=r"\d{11}", keyword="手机号",
                     level="high", right_boundary=BoundaryMode.TOKEN),
        ]
        correction = svc.correct_rules(corrected)
        thread.join()

        # 在途任务仍按旧口径完成，且登记处保留了旧版本
        assert correction["previous_version"] == pinned_version
        assert correction["rules_version"] != pinned_version
        assert RULE_REGISTRY.get(pinned_version) is not None
        assert holder["rule_set_version"] == pinned_version
        assert "error" not in holder["result"]
        assert holder["result"]["chunks"] == 3

    def test_resubmit_compares_old_and_new_model_inputs(self):
        chunk_set_id = "MASK-RESUBMIT"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        vec_svc = VectorEmbedService()

        # 第一次提交：当前确认口径（含守卫边界的编号规则）
        run1 = vec_svc.embed_masked_artifacts(chunk_set_id, run_id="RUN-OLD")
        assert run1["records_written"] == 3
        old_inputs = {r["chunk_id"]: r for r in vec_svc.read_index_inputs(chunk_set_id)}
        old_rules_version = run1["verifications"][0]["rules_version"]

        # 规则更正后重新脱敏并重新提交
        corrected = [
            MaskRule(rule_id="R_NAME", pattern=r"李明|王芳|张伟|刘洋", keyword="人名",
                     level="high"),
            MaskRule(rule_id="R_RECORD_ID", pattern=r"编号[A-Z]\d{4}",
                     keyword="档案编号", level="high"),
            MaskRule(rule_id="R_PHONE", pattern=r"\d{11}", keyword="手机号",
                     level="high", right_boundary=BoundaryMode.TOKEN),
        ]
        svc.correct_rules(corrected)
        summary2 = svc.mask_chunk_set(chunk_set_id)
        run2 = vec_svc.embed_masked_artifacts(chunk_set_id, run_id="RUN-NEW")
        assert run2["records_written"] == 3
        new_inputs = {r["chunk_id"]: r for r in vec_svc.read_index_inputs(chunk_set_id)}

        # 新旧模型输入的命中明细比对
        c1_old = old_inputs[f"{chunk_set_id}-C1"]
        c1_new = new_inputs[f"{chunk_set_id}-C1"]
        assert c1_old["source"]["rules_version"] == old_rules_version
        assert c1_new["source"]["rules_version"] == summary2["rules_version"]
        assert c1_old["source"]["rules_fingerprint"] != c1_new["source"]["rules_fingerprint"]
        # 旧口径编号规则守卫吸收人名：单遮盖区间；新口径精确边界：人名与编号各自成段
        assert c1_old["source"]["span_count"] == 1
        assert c1_new["source"]["span_count"] == 2
        # 无命中分块在两版口径下命中明细均为空，输入保持一致
        c3_old = old_inputs[f"{chunk_set_id}-C3"]
        c3_new = new_inputs[f"{chunk_set_id}-C3"]
        assert c3_old["source"]["hit_count"] == c3_new["source"]["hit_count"] == 0
        # 所有索引输入均为可消费脱敏文本，且嵌入只由脱敏文本决定
        for record in new_inputs.values():
            assert record["source"]["kind"] == "masked_chunk"
            assert record["input_text_length"] > 0

    def test_embed_never_consumes_raw_chunks(self):
        chunk_set_id = "MASK-NO-RAW"
        _write_chunks(chunk_set_id)
        svc = SensitiveFilterService()
        svc.mask_chunk_set(chunk_set_id)
        result = VectorEmbedService().embed_masked_artifacts(chunk_set_id)
        assert "error" not in result
        index_path = DATA_ROOT / "vectors" / f"{chunk_set_id}_masked_index.jsonl"
        raw = index_path.read_text(encoding="utf-8")
        # 索引产物中不得出现任何原始命中内容
        for leaked in ("李明", "编号A1234", "13800138000"):
            assert leaked not in raw
