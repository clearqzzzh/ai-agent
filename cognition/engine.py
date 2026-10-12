"""
RAG 知识库模块 v2 — ChromaDB 向量检索 + TF-IDF 降级
====================================================
- 主检索：sentence-transformers + ChromaDB（持久化向量库，支持 MMR 多样性检索）
- 兜底：TF-IDF 关键词匹配（embedding 模型未就绪时自动降级）
- 支持 docs/ 目录下 .txt/.md 文件的自动加载与索引
- ChromaDB 自动持久化，重启免重建

v1 → v2 变更：
- numpy + pickle → ChromaDB PersistentClient
- 纯 top-k → 支持 MMR（Maximal Marginal Relevance）多样性检索
- 手动余弦相似度计算 → ChromaDB 内置 ANN 检索
"""

import re
import uuid
from pathlib import Path

import jieba
from loguru import logger
from rank_bm25 import BM25Okapi

# ── 配置 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent          # agent-service/
DOCS_DIR = Path(__file__).resolve().parent / "docs"        # cognition/docs/
CACHE_DIR = BASE_DIR / "cache"
CHROMA_DIR = CACHE_DIR / "chroma_db"                       # ChromaDB 持久化目录

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 3
MMR_LAMBDA = 0.5          # MMR 多样性参数：1.0=纯相关，0.0=纯多样

# ── 混合检索权重 ──────────────────────────────────────────
HYBRID_VECTOR_WEIGHT = 0.6   # 向量检索权重
HYBRID_BM25_WEIGHT = 0.4     # BM25 关键词检索权重

COLLECTION_NAME = "pethome_knowledge"

# ── embedding 模型（懒加载，本地路径）────────────────────
_model = None
_MODEL_PATH = "shibing624/text2vec-base-chinese"

# ── ChromaDB 运行时 ──────────────────────────────────────
_chroma_client = None
_chroma_collection = None

# ── TF-IDF 降级状态 ──────────────────────────────────────
_tfidf_ready = False


# ══════════════════════════════════════════════════════════
#  预加载（供 main.py startup 调用）
# ══════════════════════════════════════════════════════════

def preload_model():
    """预加载 embedding 模型 + 构建 ChromaDB 索引，避免首次请求阻塞"""
    _get_model()
    build_index()


def _get_model():
    """懒加载 embedding 模型，失败返回 None → 自动降级 TF-IDF"""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"正在加载本地 embedding 模型: {_MODEL_PATH} ...")
            _model = SentenceTransformer(_MODEL_PATH)
            dim = _model.get_embedding_dimension()
            logger.info(f"Embedding 模型就绪，维度: {dim}")
        except Exception as e:
            logger.warning(f"Embedding 模型加载失败: {e}，将使用 TF-IDF 兜底")
            _model = False
    return _model if _model is not False else None


# ══════════════════════════════════════════════════════════
#  ChromaDB 初始化
# ══════════════════════════════════════════════════════════

class _SentenceTransformerEF:
    """
    ChromaDB 自定义 Embedding Function 适配器。
    用 sentence-transformers 本地模型生成向量，替代 ChromaDB 默认的 ONNX 模型。
    """

    name = "sentence_transformers_local"

    def __init__(self, model):
        self._model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(input, show_progress_bar=False)
        return embeddings.tolist()


def _get_chroma_collection():
    """获取或创建 ChromaDB collection（单例，不重复销毁）"""
    global _chroma_client, _chroma_collection

    if _chroma_collection is not None:
        return _chroma_collection

    try:
        import chromadb
        from chromadb.config import Settings

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        # 不传 embedding_function —— 我们手动算向量、手动传 embeddings
        # 这样避免 ChromaDB 封装 SentenceTransformer 时的各种兼容问题
        existing = _chroma_client.list_collections()
        if COLLECTION_NAME in [c.name for c in existing]:
            _chroma_collection = _chroma_client.get_collection(COLLECTION_NAME)
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' 已加载，"
                        f"共 {_chroma_collection.count()} 个文档")
        else:
            _chroma_collection = _chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' 已创建（空）")

        return _chroma_collection

    except Exception as e:
        logger.error(f"ChromaDB 初始化失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════
#  文本分块
# ══════════════════════════════════════════════════════════

def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    chunks: list[str] = []
    cur = ""
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


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = _split_paragraphs(text)
    chunks: list[str] = []
    buf = ""
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


# ══════════════════════════════════════════════════════════
#  文档加载（.txt / .md / .pdf / .docx / .html）
# ══════════════════════════════════════════════════════════

def _load_documents() -> list[dict[str, str]]:
    """加载 docs/ 目录下所有支持格式的文档（.txt/.md/.pdf/.docx/.html）"""
    from cognition.document_loader import load_documents
    return load_documents(DOCS_DIR)


# ══════════════════════════════════════════════════════════
#  索引构建
# ══════════════════════════════════════════════════════════

def build_index(force_rebuild: bool = False) -> int:
    """
    构建 ChromaDB 向量索引。
    - 手动用 sentence-transformers 算 embeddings，显式传入 collection.add()
    - ChromaDB 自动持久化到 cache/chroma_db/
    - 缓存命中（collection 已有数据）则跳过重建
    """
    global _tfidf_ready

    collection = _get_chroma_collection()
    if collection is None:
        logger.warning("ChromaDB 不可用，跳过索引构建")
        return 0

    # 已有数据且不需要强制重建 → 直接使用
    existing_count = collection.count()
    if not force_rebuild and existing_count > 0:
        logger.info(f"ChromaDB 索引缓存命中，共 {existing_count} 个 chunk")
        return existing_count

    # 强制重建：先清空 collection
    if existing_count > 0:
        try:
            collection.delete(ids=collection.get()["ids"])
            logger.info(f"已清空 collection（{existing_count} 个旧 chunk）")
        except Exception as e:
            logger.warning(f"清空 collection 失败: {e}")

    # 加载文档并分块
    documents = _load_documents()
    if not documents:
        return 0

    chunks: list[str] = []
    sources: list[str] = []
    ids: list[str] = []

    for doc in documents:
        for i, ch in enumerate(_chunk_text(doc["content"])):
            chunks.append(ch)
            sources.append(doc["source"])
            ids.append(f"{doc['source']}_chunk_{i}")

    if not chunks:
        return 0

    # 手动计算 embeddings
    model = _get_model()
    if model:
        logger.info(f"正在计算 {len(chunks)} 个 chunk 的 embeddings ...")
        embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    else:
        # 模型不可用，用零向量占位（查询时会走 TF-IDF 降级）
        logger.warning("Embedding 模型不可，用零向量占位")
        embeddings = [[0.0] * 384] * len(chunks)

    # 批量写入 ChromaDB（显式传入 embeddings，不依赖 ChromaDB 的 embedding function）
    BATCH_SIZE = 100
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.add(
            documents=chunks[start:end],
            metadatas=[{"source": s} for s in sources[start:end]],
            ids=ids[start:end],
            embeddings=embeddings[start:end],
        )

    total = collection.count()
    tag = "Embedding (dense)" if model else "Zero vectors (TF-IDF fallback)"
    logger.info(f"ChromaDB 索引构建完成: {tag}，共 {total} 个 chunk")

    _tfidf_ready = False  # 重置 TF-IDF 状态
    return total


# ══════════════════════════════════════════════════════════
#  MMR 多样性检索
# ══════════════════════════════════════════════════════════

def _mmr_select(query_embedding: list[float], candidate_embeddings: list[list[float]],
                candidate_ids: list[str], candidate_docs: list[str],
                candidate_metas: list[dict], top_k: int,
                lambda_param: float = MMR_LAMBDA) -> list[int]:
    """
    Maximal Marginal Relevance 选择算法。
    在相关性和多样性之间取平衡，避免返回大量语义重复的 chunk。

    lambda_param:
        1.0 → 纯相关性（退化为 top-k）
        0.0 → 纯多样性
        0.5 → 平衡（默认）
    """
    import numpy as np

    if not candidate_embeddings:
        return []

    q = np.array(query_embedding)
    cands = np.array(candidate_embeddings)

    # 查询与各候选的余弦相似度（ChromaDB 用 cosine distance，sim = 1 - dist）
    q_norm = np.linalg.norm(q)
    c_norms = np.linalg.norm(cands, axis=1)
    denom = c_norms * q_norm
    denom[denom == 0] = 1e-10
    similarities = np.dot(cands, q) / denom  # shape: (n_candidates,)

    selected: list[int] = []
    remaining = set(range(len(candidate_embeddings)))

    # 第一步：选最相关的
    best = int(np.argmax(similarities))
    selected.append(best)
    remaining.remove(best)

    # 后续：MMR 贪心选择
    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_idx = -1
        for idx in remaining:
            # 与查询的相关性
            relevance = similarities[idx]
            # 与已选文档的最大相似度（越不相似越好）
            max_sim_to_selected = max(
                float(np.dot(cands[idx], cands[s]) / (c_norms[idx] * c_norms[s] + 1e-10))
                for s in selected
            )
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


# ══════════════════════════════════════════════════════════
#  Query 改写（LLM 口语→搜索关键词）
# ══════════════════════════════════════════════════════════

async def rewrite_query(query: str) -> str:
    """
    用 LLM 将口语化查询改写为检索友好的关键词。
    例: "我家猫吐了怎么办" → "猫呕吐 原因 处理 护理"
    失败时返回原始 query（不阻断流程）。
    """
    try:
        import httpx
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key":
            return query

        resp = await httpx.AsyncClient(timeout=5).post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content":
                     "你是查询改写助手。将用户的口语化问题改写为3-5个检索关键词，"
                     "用空格分隔。只输出关键词，不要解释。"},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 50,
                "temperature": 0.3,
            },
        )
        rewritten = resp.json()["choices"][0]["message"]["content"].strip()
        if rewritten:
            logger.info(f"Query改写: '{query}' → '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        logger.warning(f"Query改写失败，使用原始查询: {e}")
        return query


def _rewrite_query_sync(query: str) -> str:
    """rewrite_query 的同步版本，用 httpx 同步客户端，避免 async 死锁。"""
    try:
        import httpx
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key":
            return query

        resp = httpx.Client(timeout=5).post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content":
                     "你是查询改写助手。将用户的口语化问题改写为3-5个检索关键词，"
                     "用空格分隔。只输出关键词，不要解释。"},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 50,
                "temperature": 0.3,
            },
        )
        rewritten = resp.json()["choices"][0]["message"]["content"].strip()
        if rewritten:
            logger.info(f"Query改写: '{query}' → '{rewritten}'")
            return rewritten
        return query
    except Exception as e:
        logger.warning(f"Query改写失败，使用原始查询: {e}")
        return query


# ══════════════════════════════════════════════════════════
#  BM25 关键词检索
# ══════════════════════════════════════════════════════════

_bm25_cache: dict | None = None   # {"index": BM25Okapi, "chunks": [...], "sources": [...]}


def _get_bm25_index() -> dict | None:
    """懒加载 BM25 索引（jieba 分词），首次调用时构建并缓存。"""
    global _bm25_cache
    if _bm25_cache is not None:
        return _bm25_cache

    try:
        documents = _load_documents()
        if not documents:
            return None

        chunks: list[str] = []
        sources: list[str] = []
        for doc in documents:
            for ch in _chunk_text(doc["content"]):
                chunks.append(ch)
                sources.append(doc["source"])

        if not chunks:
            return None

        # jieba 分词
        tokenized = [list(jieba.cut(ch)) for ch in chunks]
        index = BM25Okapi(tokenized)

        _bm25_cache = {"index": index, "chunks": chunks, "sources": sources}
        logger.info(f"BM25 索引构建完成，共 {len(chunks)} 个 chunk")
        return _bm25_cache

    except Exception as e:
        logger.error(f"BM25 索引构建失败: {e}")
        return None


def _search_bm25(query: str, top_k: int) -> list[dict]:
    """BM25 关键词检索（jieba 中文分词）。"""
    cache = _get_bm25_index()
    if cache is None:
        return []

    try:
        tokens = list(jieba.cut(query))
        scores = cache["index"].get_scores(tokens)

        # 取 top_k 个最高分
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": cache["chunks"][idx],
                    "source": cache["sources"][idx],
                    "score": round(float(scores[idx]), 4),
                })
        return results
    except Exception as e:
        logger.error(f"BM25 检索失败: {e}")
        return []


# ══════════════════════════════════════════════════════════
#  分数融合（向量 + BM25 加权合并）
# ══════════════════════════════════════════════════════════

def _fuse_scores(vector_results: list[dict], bm25_results: list[dict],
                 top_k: int) -> list[dict]:
    """
    将向量检索和 BM25 检索的结果按权重融合。
    同一 chunk（按 content 匹配）的分数加权合并。
    """
    fused: dict[str, dict] = {}  # key = content 前 100 字符

    for r in vector_results:
        key = r["content"][:100]
        fused[key] = {
            "content": r["content"],
            "source": r["source"],
            "score": r["score"] * HYBRID_VECTOR_WEIGHT,
        }

    for r in bm25_results:
        key = r["content"][:100]
        if key in fused:
            fused[key]["score"] += r["score"] * HYBRID_BM25_WEIGHT
            # 保留向量检索的 source（更可靠）
        else:
            fused[key] = {
                "content": r["content"],
                "source": r["source"],
                "score": r["score"] * HYBRID_BM25_WEIGHT,
            }

    # 按融合分数降序排列
    sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]


# ══════════════════════════════════════════════════════════
#  Rerank 重排序（embedding 余弦相似度精排）
# ══════════════════════════════════════════════════════════

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    用 embedding 模型对候选结果重新打分（精排）。
    计算 query 与各 chunk 的余弦相似度，替换原始分数。
    """
    model = _get_model()
    if model is None or not candidates:
        return candidates[:top_k]

    try:
        q_emb = model.encode([query], show_progress_bar=False)[0]
        texts = [c["content"] for c in candidates]
        c_embs = model.encode(texts, show_progress_bar=False)

        import numpy as np
        q_norm = np.linalg.norm(q_emb)
        for i, c_emb in enumerate(c_embs):
            c_norm = np.linalg.norm(c_emb)
            denom = c_norm * q_norm
            if denom == 0:
                denom = 1e-10
            cos_sim = float(np.dot(c_emb, q_emb) / denom)
            candidates[i]["score"] = round(cos_sim, 4)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    except Exception as e:
        logger.warning(f"Rerank 失败，保持原排序: {e}")
        return candidates[:top_k]


# ══════════════════════════════════════════════════════════
#  向量检索（ChromaDB，提取为独立函数供混合检索复用）
# ══════════════════════════════════════════════════════════

def _vector_search(query: str, top_k: int, use_mmr: bool = True) -> list[dict]:
    """
    纯向量检索（ChromaDB + sentence-transformers + MMR）。
    不可用时返回空列表（不降级 TF-IDF，由调用方决定降级策略）。
    """
    collection = _get_chroma_collection()
    if collection is None or collection.count() == 0:
        return []

    model = _get_model()
    if model is None:
        return []

    try:
        q_vec = model.encode([query], show_progress_bar=False).tolist()[0]

        fetch_k = top_k * 4 if use_mmr else top_k
        fetch_k = min(fetch_k, collection.count())

        include_fields = ["documents", "metadatas", "distances"]
        if use_mmr:
            include_fields.append("embeddings")

        results = collection.query(
            query_embeddings=[q_vec],
            n_results=fetch_k,
            include=include_fields,
        )

        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        embs = None
        if use_mmr:
            raw_embs = results.get("embeddings")
            if raw_embs and len(raw_embs) > 0:
                first = raw_embs[0]
                if first is not None:
                    embs = [list(e) for e in first] if hasattr(first, '__len__') else None

        if not docs:
            return []

        if use_mmr and embs and len(docs) > top_k:
            selected_indices = _mmr_select(
                query_embedding=q_vec,
                candidate_embeddings=embs,
                candidate_ids=(results.get("ids") or [[]])[0],
                candidate_docs=docs,
                candidate_metas=metas,
                top_k=top_k,
                lambda_param=MMR_LAMBDA,
            )
            output = []
            for idx in selected_indices:
                score = round(1.0 - (dists[idx] if idx < len(dists) else 1.0), 4)
                meta = metas[idx] if idx < len(metas) else {}
                output.append({
                    "content": docs[idx],
                    "source": meta.get("source", "unknown") if meta else "unknown",
                    "score": score,
                })
            return output
        else:
            output = []
            for i in range(min(top_k, len(docs))):
                dist = dists[i] if i < len(dists) else 1.0
                score = round(1.0 - dist, 4)
                if score > 0:
                    meta = metas[i] if i < len(metas) else {}
                    output.append({
                        "content": docs[i],
                        "source": meta.get("source", "unknown") if meta else "unknown",
                        "score": score,
                    })
            return output

    except Exception as e:
        import traceback
        logger.error(f"ChromaDB 向量检索异常: {e}")
        logger.debug(traceback.format_exc())
        return []


# ══════════════════════════════════════════════════════════
#  检索主入口（混合检索：Query改写 + 向量/BM25融合 + Rerank）
# ══════════════════════════════════════════════════════════

def search_knowledge(query: str, top_k: int = TOP_K_DEFAULT,
                     use_mmr: bool = True, use_hybrid: bool = True,
                     use_rewrite: bool = True) -> list[dict]:
    """
    混合检索入口。完整流程:
      1. Query 改写（LLM 口语→关键词）
      2. 双路召回: 向量检索(ChromaDB) + BM25关键词检索
      3. 加权融合
      4. Rerank 精排

    降级链:
      混合检索 → 纯向量检索 → TF-IDF

    参数:
        query:       查询文本
        top_k:       返回结果数
        use_mmr:     向量检索是否启用 MMR 多样性
        use_hybrid:  是否启用混合检索（向量+BM25），False 则纯向量
        use_rewrite: 是否启用 Query 改写

    返回:
        [{"content": ..., "source": ..., "score": ...}, ...]
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    # ── Step 1: Query 改写 + 向量检索 并行执行 ─
    t_total = _time.perf_counter()
    fetch_k = top_k * 3  # 多召回一些用于融合+rerank

    if use_hybrid and use_rewrite:
        # 并行：向量检索用原始 query（不需要改写），改写只影响 BM25
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_rewrite = pool.submit(_rewrite_query_sync, query)
            future_vector = pool.submit(_vector_search, query, fetch_k, use_mmr)

            search_query = future_rewrite.result()
            vector_results = future_vector.result()

        t_rewrite = None  # 无法单独计时（并行）
    elif use_hybrid:
        search_query = query
        vector_results = _vector_search(query, fetch_k, use_mmr)
    else:
        search_query = query
        if use_rewrite:
            search_query = _rewrite_query_sync(query)
        vector_results = _vector_search(search_query, fetch_k, use_mmr)

    # ── Step 2: BM25 检索（用改写后的 query）──
    if use_hybrid:
        t0 = _time.perf_counter()
        bm25_results = _search_bm25(search_query, fetch_k)
        t_bm25 = (_time.perf_counter() - t0) * 1000

        # 融合
        if vector_results and bm25_results:
            candidates = _fuse_scores(vector_results, bm25_results, fetch_k)
            logger.info(f"混合检索: 向量{len(vector_results)}条 + BM25{len(bm25_results)}条({t_bm25:.0f}ms) → 融合{len(candidates)}条")
        elif vector_results:
            candidates = vector_results
            logger.info(f"BM25无结果，仅使用向量检索: {len(vector_results)}条")
        elif bm25_results:
            candidates = bm25_results
            logger.info(f"向量检索无结果，仅使用BM25: {len(bm25_results)}条")
        else:
            logger.warning("混合检索两路皆空，降级 TF-IDF")
            return _search_tfidf(query, top_k)
    else:
        if vector_results:
            candidates = vector_results
        else:
            return _search_tfidf(query, top_k)

    # ── Step 3: Rerank 精排（限制候选数，减少 encode 开销）──
    RERANK_CANDIDATES = min(len(candidates), 6)  # 最多 rerank 6 条
    t0 = _time.perf_counter()
    results = _rerank(search_query, candidates[:RERANK_CANDIDATES], top_k)
    t_rerank = (_time.perf_counter() - t0) * 1000

    # 确保分数 > 0
    results = [r for r in results if r.get("score", 0) > 0]

    t_total_ms = (_time.perf_counter() - t_total) * 1000
    logger.info(f"[耗时] 检索总耗时: {t_total_ms:.0f}ms (rerank: {t_rerank:.0f}ms, 候选: {RERANK_CANDIDATES}条)")

    return results


# ══════════════════════════════════════════════════════════
#  医疗知识库检索（仅检索 pet_medical.md）
# ══════════════════════════════════════════════════════════

def search_medical_knowledge(query: str, top_k: int = TOP_K_DEFAULT,
                              use_mmr: bool = True) -> list[dict]:
    """
    医疗知识语义检索。复用 search_knowledge，但只返回 pet_medical.md 的结果。
    用于医疗Agent的症状分析、护理建议、就医指引。

    返回:
        [{"content": ..., "source": ..., "score": ...}, ...]
    """
    results = search_knowledge(query, top_k=top_k * 2, use_mmr=use_mmr)
    medical_results = [r for r in results if r.get("source") == "pet_medical.md"]
    return medical_results[:top_k]


# ══════════════════════════════════════════════════════════
#  TF-IDF 降级检索（内存计算）
# ══════════════════════════════════════════════════════════

def _search_tfidf(query: str, top_k: int) -> list[dict]:
    """TF-IDF 降级检索。当 embedding 模型或 ChromaDB 不可用时使用。"""
    global _tfidf_ready
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = _load_documents()
        if not documents:
            return []

        chunks: list[str] = []
        sources: list[str] = []
        for doc in documents:
            for ch in _chunk_text(doc["content"]):
                chunks.append(ch)
                sources.append(doc["source"])

        if not chunks:
            return []

        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), max_features=2048)
        tfidf_mat = vec.fit_transform(chunks).toarray().astype(np.float32)
        q_vec = vec.transform([query]).toarray().astype(np.float32)

        # 余弦相似度
        q = q_vec.ravel()
        q_norm = np.linalg.norm(q)
        d_norm = np.linalg.norm(tfidf_mat, axis=1)
        denom = d_norm * q_norm
        denom[denom == 0] = 1e-10
        scores = np.dot(tfidf_mat, q) / denom

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": chunks[idx],
                    "source": sources[idx],
                    "score": round(float(scores[idx]), 4),
                })
        return results
    except Exception as e:
        logger.error(f"TF-IDF 降级检索失败: {e}")
        return []


# ══════════════════════════════════════════════════════════
#  启动时自动建索引
# ══════════════════════════════════════════════════════════

try:
    cnt = build_index()
    model = _get_model()
    tag = "ChromaDB + Embedding" if model else "ChromaDB (default EF)"
    logger.info(f"RAG 索引就绪: {tag}，共 {cnt} 个 chunk")
except Exception as e:
    logger.error(f"RAG 索引构建失败: {e}")
