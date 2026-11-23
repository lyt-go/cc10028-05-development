"""向量嵌入服务：文本分块的向量嵌入模型选择、批次写入与向量索引地址登记。

从分块产物读取文本分块，执行向量嵌入计算并将向量写入索引文件。

安全约束（送嵌入链路）：
``embed_masked_artifacts`` 只读取脱敏分块作为计算输入，并在写入每条带来源的
数据索引前，用脱敏凭据核对实际输入；原始分块文本不进入嵌入计算。
"""
from __future__ import annotations

import time
import hashlib
from typing import Any

from app.services.base import BaseModuleService
from app.services.masking import MASK_TOKEN
from app.store import store


class VectorEmbedService(BaseModuleService):
    MODULE = "vector_embed"
    ENTITY = "向量化任务"
    OPERATION_TERM = "写入"
    LIST_FIELDS = ["任务编号", "分块编号", "嵌入模型", "向量维度", "批次大小", "向量条数", "索引地址", "模型版本"]
    REQUIRED_FIELDS = ["任务编号", "分块编号", "嵌入模型"]
    STATUS_ORDER = ["草稿", "待嵌入", "嵌入中", "已完成"]
    ACTION_RULES = {"启动嵌入": "待嵌入", "标记完成": "已完成", "标记废弃": "已完成"}
    NEGATIVE_ACTIONS = []

    MODEL_PRESETS = {
        "text2vec-large": {"dimension": 1024, "batch_size": 32},
        "bge-large": {"dimension": 1024, "batch_size": 64},
        "m3e-base": {"dimension": 768, "batch_size": 32},
    }
# 数据操作：写入文本分块向量化生成向量索引数据集

    def embed_from_chunk(self, chunk_set_id: str) -> dict[str, Any]:
        """从分块产物读取文本分块，执行向量嵌入并持久化到向量索引文件。"""
        chunk_file = self._data_dir("chunks") / f"{chunk_set_id}_chunks.jsonl"
        chunks = self._read_jsonl(chunk_file)
        if not chunks:
            return {"error": f"分块产物 {chunk_set_id} 不存在或为空，请先执行分块切分"}

        model = "text2vec-large"
        model_info = self.MODEL_PRESETS.get(model, {"dimension": 1024, "batch_size": 32})
        dimension = model_info["dimension"]

        vectors = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text:
                continue
            emb = self._compute_embedding(text, dimension)
            vectors.append({
                "chunk_id": chunk.get("chunk_id"),
# 数据产物：向量索引数据集写入vectors目录持久化文件
                "doc_id": chunk.get("doc_id"),
                "vector_dim": dimension,
                "vector": emb,
                "model": model,
                "text_length": len(text),
                "index": f"vector_index/{chunk.get('chunk_id')}",
                "embedded_at": time.time(),
            })

        output_path = self._data_dir("vectors") / f"{chunk_set_id}_vectors.jsonl"
        written = self._write_jsonl(output_path, vectors)

        self._write_checkpoint(f"vector_{chunk_set_id}", {
            "chunk_set_id": chunk_set_id,
            "model": model,
            "dimension": dimension,
            "chunks_embedded": len(vectors),
            "total_dimension": len(vectors) * dimension,
        })

        return {
            "chunk_set_id": chunk_set_id,
            "model": model,
            "dimension": dimension,
            "chunks": len(vectors),
            "records_written": written,
            "output": str(output_path),
        }

    def _compute_embedding(self, text: str, dimension: int) -> list[float]:
        """模拟向量嵌入计算，基于文本哈希生成伪嵌入向量用于测试数据流。"""
        hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        vector = []
        for i in range(dimension):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) - 0.5
            if i % 3 == 0:
                value *= 1.5
            elif i % 5 == 0:
                value *= 0.8
            vector.append(round(value, 6))
        return vector

    def batch_embed_documents(self, doc_ids: list[str]) -> dict[str, Any]:
        """批量对多个文档执行向量嵌入，返回汇总统计。"""
        results = []
        for doc_id in doc_ids:
            try:
                r = self.embed_from_chunk(f"{doc_id}_parsed")
                if "error" not in r:
                    results.append(r)
            except Exception:
                pass
        return {
            "documents": len(doc_ids),
            "success": len(results),
            "total_vectors": sum(r.get("chunks", 0) for r in results),
        }

    # ==================================================================
    # 脱敏分块 → 向量索引：只以可消费的脱敏文本为计算输入
    # ==================================================================

    def embed_masked_artifacts(
        self,
        chunk_set_id: str,
        *,
        run_id: str | None = None,
        rules_version: str | None = None,
        model: str = "bge-large",
    ) -> dict[str, Any]:
        """读取送嵌入的脱敏分块，凭据核对通过后写出带来源的数据索引。

        严格保证：
        - 输入只来自 ``data/masked/`` 的脱敏分块，不回读原始分块；
        - 每条索引写入前由敏感词过滤服务按手上的脱敏凭据核对实际输入；
        - 任一条核对失败则拒绝整批写入，避免未授权文本进入向量索引。
        """
        from app.services.sensitive_filter import SensitiveFilterService

        filter_svc = SensitiveFilterService()
        masked_file = self._data_dir("masked") / f"{chunk_set_id}_masked_chunks.jsonl"
        masked_chunks = self._read_jsonl(masked_file)
        if not masked_chunks:
            return {"error": f"脱敏分块产物 {chunk_set_id} 不存在，请先执行送嵌入脱敏"}

        model_info = self.MODEL_PRESETS.get(model, {"dimension": 1024, "batch_size": 64})
        dimension = model_info["dimension"]
        run_id = run_id or f"RUN-{self._compute_checksum(chunk_set_id + str(time.time_ns()))}"

        index_records: list[dict[str, Any]] = []
        verifications: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for chunk in masked_chunks:
            chunk_id = str(chunk.get("chunk_id"))
            masked_text = chunk.get("masked_text", "")
            credential = filter_svc.load_credential(chunk_set_id, chunk_id)
            check = filter_svc.verify_embedding_input(
                credential,
                masked_text,
                rules_version=rules_version,
                chunk_version=chunk.get("chunk_version"),
            )
            verifications.append({"chunk_id": chunk_id, **check})
            if not check["passed"]:
                rejected.append({"chunk_id": chunk_id, "violations": check["violations"]})
                continue

            # 唯一计算输入：脱敏文本。嵌入函数接触不到原始分块。
            vector = self._compute_embedding(masked_text, dimension)
            index_records.append({
                "run_id": run_id,
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "index": f"vector_index/{chunk_id}",
                "vector_dim": dimension,
                "vector": vector,
                "model": model,
                # 来源与口径：索引可追溯到脱敏分块版本、规则版本与凭据。
                "source": {
                    "kind": "masked_chunk",
                    "chunk_version": chunk.get("chunk_version"),
                    "rules_version": chunk.get("rules_version"),
                    "rules_fingerprint": chunk.get("rules_fingerprint"),
                    "credential_id": chunk.get("credential_id"),
                    "masked_checksum": chunk.get("masked_checksum"),
                    "hit_count": chunk.get("hit_count"),
                    "span_count": chunk.get("span_count"),
                    "mask_token": MASK_TOKEN,
                },
                "input_text_length": len(masked_text),
                "embedded_at": time.time(),
            })

        if rejected:
            return {
                "error": "脱敏凭据核对未通过，已拒绝整批索引写入",
                "chunk_set_id": chunk_set_id,
                "run_id": run_id,
                "rejected": rejected,
                "verified": len(verifications),
                "written": 0,
            }

        output_path = self._data_dir("vectors") / f"{chunk_set_id}_masked_index.jsonl"
        # 同一分块集的重新提交覆盖写，保证回读到的始终是最近一次模型输入。
        output_path.write_text(
            "\n".join(self._json_line(rec) for rec in index_records) + "\n",
            encoding="utf-8",
        )
        written = len(index_records)
        self._write_checkpoint(f"masked_vector_{chunk_set_id}", {
            "chunk_set_id": chunk_set_id,
            "run_id": run_id,
            "model": model,
            "dimension": dimension,
            "rules_version": rules_version or "pinned-by-credential",
            "chunks_embedded": len(index_records),
            "all_inputs_verified": True,
        })
        return {
            "chunk_set_id": chunk_set_id,
            "run_id": run_id,
            "model": model,
            "dimension": dimension,
            "chunks": len(index_records),
            "records_written": written,
            "verifications": verifications,
            "output": str(output_path),
        }

    def read_index_inputs(self, chunk_set_id: str) -> list[dict[str, Any]]:
        """回读索引产物中登记的实际模型输入来源，供新旧请求命中明细比对。"""
        path = self._data_dir("vectors") / f"{chunk_set_id}_masked_index.jsonl"
        return self._read_jsonl(path)

    @staticmethod
    def _json_line(record: dict[str, Any]) -> str:
        import json
        return json.dumps(record, ensure_ascii=False, default=str)