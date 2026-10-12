"""
Chunking 策略对比测试
=====================
用法: cd agent-service && python tests/compare_chunking.py

对比不同 chunk 大小和重叠策略对检索质量的影响。
基于 faq_test_set.json 评估 Source Accuracy 和 Keyword Recall。

测试矩阵:
- chunk_size: [300, 500, 800]
- chunk_overlap: [0, 50, 100]
"""

import json
import sys
import time
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_SET_PATH = Path(__file__).resolve().parent / "faq_test_set.json"
DOCS_DIR = PROJECT_ROOT / "cognition" / "docs"


# ── 分块函数（与 engine.py 保持一致）─────────────────────
def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    chunks, cur = [], ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(cur) + len(sent) + 1 <= size:
            cur = (cur + sent) if cur else sent
        else:
            if cur:
                chunks.append(cur)
            if len(sent) > size:
                step = size - overlap
                chunks.extend(sent[i:i + size] for i in range(0, len(sent), step))
            else:
                cur = sent
    if cur:
        chunks.append(cur)
    return chunks


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = _split_paragraphs(text)
    chunks, buf = [], ""
    for para in paragraphs:
        if len(buf) + len(para) + 1 <= size:
            buf = (buf + "\n" + para).strip() if buf else para
        else:
            if buf:
                chunks.append(buf)
            if len(para) > size:
                chunks.extend(_split_long_text(para, size, overlap))
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            out.append(chunks[i - 1][-overlap:] + " " + chunks[i])
        return out
    return chunks


# ── 用指定参数构建临时索引并评估 ──────────────────────────
def evaluate_with_params(chunk_size: int, chunk_overlap: int, top_k: int = 3) -> dict:
    """用指定的 chunking 参数构建临时索引，评估检索质量。"""
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import chromadb
    from chromadb.config import Settings

    model_path = str(PROJECT_ROOT / "models" / "embedding")
    model = SentenceTransformer(model_path)

    # 加载文档
    docs = []
    for fpath in sorted(DOCS_DIR.iterdir()):
        if fpath.suffix in (".txt", ".md"):
            docs.append({
                "source": fpath.name,
                "content": fpath.read_text(encoding="utf-8"),
            })

    # 分块
    chunks, sources, ids = [], [], []
    for doc in docs:
        for i, ch in enumerate(_chunk_text(doc["content"], chunk_size, chunk_overlap)):
            chunks.append(ch)
            sources.append(doc["source"])
            ids.append(f"{doc['source']}_s{chunk_size}_o{chunk_overlap}_c{i}")

    if not chunks:
        return {"error": "no chunks"}

    # 构建临时 ChromaDB collection
    cache_dir = PROJECT_ROOT / "cache"
    cache_dir.mkdir(exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(cache_dir / "chroma_db_eval"),
        settings=Settings(anonymized_telemetry=False),
    )

    # 清理同名 collection
    col_name = f"eval_s{chunk_size}_o{chunk_overlap}"
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    class _EF:
        def __init__(self, m): self._m = m
        def __call__(self, input): return self._m.encode(input, show_progress_bar=False).tolist()

    collection = client.get_or_create_collection(
        name=col_name,
        embedding_function=_EF(model),
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 100
    for start in range(0, len(chunks), BATCH):
        end = start + BATCH
        collection.add(
            documents=chunks[start:end],
            metadatas=[{"source": s} for s in sources[start:end]],
            ids=ids[start:end],
        )

    # 评估
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    source_hits = 0
    keyword_total = 0.0
    latencies = []

    for case in test_set:
        q = case["question"]
        expected = case["expected_source"]
        keywords = case["expected_keywords"]

        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[model.encode([q], show_progress_bar=False).tolist()[0]],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas"],
        )
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        ret_sources = [m["source"] for m in results["metadatas"][0]]
        if expected in ret_sources:
            source_hits += 1

        all_text = " ".join(results["documents"][0])
        if keywords:
            hits = sum(1 for kw in keywords if kw in all_text)
            keyword_total += hits / len(keywords)

    total = len(test_set)
    # 清理临时 collection
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "source_accuracy": round(source_hits / total, 4),
        "avg_keyword_recall": round(keyword_total / total, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
    }


def main():
    # 测试矩阵
    configs = [
        (300, 0), (300, 50), (300, 100),
        (500, 0), (500, 50), (500, 100),
        (800, 0), (800, 50), (800, 100),
    ]

    print("Chunking 策略对比测试")
    print("=" * 70)
    print(f"{'size':>5} {'overlap':>7} {'chunks':>6} {'来源准确率':>10} "
          f"{'关键词召回':>10} {'平均延迟':>10}")
    print("-" * 70)

    results = []
    for size, overlap in configs:
        print(f"  测试 size={size}, overlap={overlap} ...", end=" ", flush=True)
        r = evaluate_with_params(size, overlap)
        results.append(r)
        print(f"chunks={r['total_chunks']:>4}  "
              f"准确率={r['source_accuracy']:.1%}  "
              f"召回={r['avg_keyword_recall']:.1%}  "
              f"延迟={r['avg_latency_ms']:.0f}ms")

    # 找最优配置
    best = max(results, key=lambda x: (x["source_accuracy"], x["avg_keyword_recall"]))
    print("\n" + "=" * 70)
    print(f"最优配置: size={best['chunk_size']}, overlap={best['chunk_overlap']}")
    print(f"  来源准确率: {best['source_accuracy']:.1%}")
    print(f"  关键词召回: {best['avg_keyword_recall']:.1%}")
    print(f"  平均 chunk 数: {best['total_chunks']}")

    # 保存结果
    out_path = Path(__file__).resolve().parent / "chunking_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存: {out_path}")


if __name__ == "__main__":
    main()
