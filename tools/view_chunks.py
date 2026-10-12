"""
查看 ChromaDB 知识库中所有 chunk 的内容
用法: python tools/view_chunks.py [--limit N] [--id CHUNK_ID]

示例:
  python tools/view_chunks.py              # 查看所有 chunk 概览
  python tools/view_chunks.py --limit 5    # 只看前 5 个
  python tools/view_chunks.py --id cat_feeding.md_chunk_0  # 查看指定 chunk 全文
"""
import sys
import os
import argparse

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import chromadb

CHROMA_DIR = os.path.join(PROJECT_ROOT, "cache", "chroma_db")
COLLECTION_NAME = "pethome_knowledge"


def view_chunks(limit=None, chunk_id=None):
    """查看知识库 chunk 内容"""
    if not os.path.exists(CHROMA_DIR):
        print(f"❌ ChromaDB 目录不存在: {CHROMA_DIR}")
        print("   请先运行 build_index() 构建知识库")
        return

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 检查 collection 是否存在
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in existing:
        print(f"❌ Collection '{COLLECTION_NAME}' 不存在")
        print(f"   当前可用: {existing}")
        return

    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    print(f"📚 Collection: {COLLECTION_NAME}")
    print(f"📊 总 chunk 数: {total}")
    print("=" * 80)

    # 查看指定 chunk
    if chunk_id:
        result = collection.get(ids=[chunk_id])
        if result["ids"]:
            doc_text = result["documents"][0]
            metadata = result["metadatas"][0] if result["metadatas"] else {}
            print(f"\n[Chunk] ID: {chunk_id}")
            if metadata:
                print(f"  元数据: {metadata}")
            print(f"\n全文内容:\n{doc_text}")
        else:
            print(f" 未找到 chunk: {chunk_id}")
            # 列出相似的 ID
            all_ids = collection.get(limit=total)["ids"]
            similar = [i for i in all_ids if chunk_id.split("_chunk")[0] in i]
            if similar:
                print(f"\n相似 ID: {similar[:10]}")
        return

    # 查看所有 chunk 概览
    get_limit = limit if limit else total
    results = collection.get(limit=get_limit)

    for i, (doc_id, doc_text) in enumerate(zip(results["ids"], results["documents"])):
        preview = doc_text[:300].replace("\n", " ")
        print(f"\n[Chunk {i+1}/{total}] ID: {doc_id}")
        print(f"  内容: {preview}...")
        print("-" * 60)

    if limit and limit < total:
        print(f"\n... 仅显示前 {limit} 个，共 {total} 个 chunk")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 ChromaDB 知识库 chunk 内容")
    parser.add_argument("--limit", type=int, default=None, help="限制显示数量")
    parser.add_argument("--id", type=str, default=None, help="查看指定 chunk ID 的全文")
    args = parser.parse_args()

    view_chunks(limit=args.limit, chunk_id=args.id)
