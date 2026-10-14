"""
RAG 知识库评估脚本
==================
评估指标:
1. Source Accuracy（来源准确率）: top-k 结果中是否包含期望的文档来源
2. Keyword Recall（关键词召回率）: 返回内容中命中期望关键词的比例
3. MRR（Mean Reciprocal Rank）: 第一个正确结果的排名倒数的均值
4. Latency（响应延迟）: 单次检索的平均/P50/P95 耗时
"""
import json
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cognition.engine import search_knowledge, build_index

# ── 配置 ──────────────────────────────────────────────────
TEST_SET_PATH = Path(__file__).resolve().parent / "faq_test_set.json"
TOP_K = 7           # 调参结果：k=7 来源100%，关键词78.3%
USE_MMR = True      # 测试 MMR 模式


def load_test_set() -> list[dict]:
    with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_single(case: dict, top_k: int = TOP_K, use_mmr: bool = USE_MMR) -> dict:
    """评估单条测试用例，返回详细结果。"""
    question = case["question"]
    expected_source = case["expected_source"]
    expected_keywords = case["expected_keywords"]

    t0 = time.perf_counter()
    results = search_knowledge(question, top_k=top_k, use_mmr=use_mmr)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 1. Source Accuracy: 期望来源是否出现在 top-k 中
    retrieved_sources = [r["source"] for r in results]
    source_hit = expected_source in retrieved_sources

    # 2. 正确来源的排名（用于 MRR）
    reciprocal_rank = 0.0
    if source_hit:
        rank = retrieved_sources.index(expected_source) + 1
        reciprocal_rank = 1.0 / rank

    # 3. Keyword Recall: 在所有返回内容中，期望关键词的命中率
    all_content = " ".join(r["content"] for r in results)
    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if kw in all_content)
        keyword_recall = hits / len(expected_keywords)
    else:
        keyword_recall = 0.0

    # 4. 命中的关键词列表（便于调试）
    hit_keywords = [kw for kw in expected_keywords if kw in all_content]
    miss_keywords = [kw for kw in expected_keywords if kw not in all_content]

    return {
        "id": case["id"],
        "category": case["category"],
        "question": question,
        "expected_source": expected_source,
        "retrieved_sources": retrieved_sources,
        "source_hit": source_hit,
        "reciprocal_rank": reciprocal_rank,
        "keyword_recall": round(keyword_recall, 4),
        "hit_keywords": hit_keywords,
        "miss_keywords": miss_keywords,
        "latency_ms": round(elapsed_ms, 2),
        "num_results": len(results),
    }


def run_evaluation(verbose: bool = True) -> dict:
    """运行完整评估，返回汇总报告。"""
    test_set = load_test_set()

    # 确保索引已构建
    build_index()

    results = []
    for case in test_set:
        r = evaluate_single(case)
        results.append(r)
        if verbose:
            status = "✓" if r["source_hit"] else "✗"
            kw_str = f"关键词 {r['keyword_recall']:.0%}"
            print(f"  [{status}] #{r['id']:02d} {r['question'][:30]:30s}  "
                  f"来源:{'✓' if r['source_hit'] else '✗'}  {kw_str}  "
                  f"{r['latency_ms']:.0f}ms")

    # ── 汇总统计 ──
    total = len(results)
    source_hits = sum(1 for r in results if r["source_hit"])
    source_accuracy = source_hits / total if total > 0 else 0

    mrr = sum(r["reciprocal_rank"] for r in results) / total if total > 0 else 0

    avg_keyword_recall = (
        sum(r["keyword_recall"] for r in results) / total if total > 0 else 0
    )

    latencies = sorted(r["latency_ms"] for r in results)
    avg_latency = sum(latencies) / len(latencies)
    p50_latency = latencies[len(latencies) // 2]
    p95_latency = latencies[int(len(latencies) * 0.95)]

    # 按类别统计
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "source_hits": 0, "keyword_sum": 0}
        categories[cat]["total"] += 1
        if r["source_hit"]:
            categories[cat]["source_hits"] += 1
        categories[cat]["keyword_sum"] += r["keyword_recall"]

    report = {
        "total_cases": total,
        "source_accuracy": round(source_accuracy, 4),
        "mrr": round(mrr, 4),
        "avg_keyword_recall": round(avg_keyword_recall, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "by_category": {
            cat: {
                "total": d["total"],
                "source_accuracy": round(d["source_hits"] / d["total"], 4) if d["total"] > 0 else 0,
                "avg_keyword_recall": round(d["keyword_sum"] / d["total"], 4) if d["total"] > 0 else 0,
            }
            for cat, d in categories.items()
        },
        "failures": [
            {
                "id": r["id"],
                "question": r["question"],
                "expected": r["expected_source"],
                "got": r["retrieved_sources"],
                "miss_keywords": r["miss_keywords"],
            }
            for r in results if not r["source_hit"]
        ],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("RAG 评估报告")
        print("=" * 60)
        print(f"测试用例数:     {total}")
        print(f"来源准确率:     {source_accuracy:.1%}  ({source_hits}/{total})")
        print(f"MRR:            {mrr:.4f}")
        print(f"关键词召回率:   {avg_keyword_recall:.1%}")
        print(f"平均延迟:       {avg_latency:.1f}ms")
        print(f"P50 延迟:       {p50_latency:.1f}ms")
        print(f"P95 延迟:       {p95_latency:.1f}ms")
        print()
        print("── 按类别 ──")
        for cat, d in report["by_category"].items():
            print(f"  {cat:12s}  准确率:{d['source_accuracy']:.0%}  "
                  f"关键词:{d['avg_keyword_recall']:.0%}  "
                  f"(n={d['total']})")

        if report["failures"]:
            print(f"\n── 来源未命中 ({len(report['failures'])}条) ──")
            for f in report["failures"]:
                print(f"  #{f['id']:02d} {f['question'][:35]}")
                print(f"       期望: {f['expected']}  实际: {f['got']}")
                if f["miss_keywords"]:
                    print(f"       缺失关键词: {f['miss_keywords']}")

    return report


def save_report(report: dict, path: str = None):
    """保存评估报告到 JSON 文件。"""
    if path is None:
        path = str(Path(__file__).resolve().parent / "eval_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--judge", action="store_true",
                        help="启用 LLM as a Judge 评测（答案质量打分）")
    parser.add_argument("--judge-limit", type=int, default=10,
                        help="LLM Judge 评测的用例数量上限（默认10）")
    args = parser.parse_args()

    print("正在初始化 RAG 索引...\n")
    report = run_evaluation(verbose=True)
    save_report(report)

    if args.judge:
        print("\n" + "=" * 60)
        print("启动 LLM as a Judge 评测...")
        print("=" * 60)
        judge_report = run_llm_judge_evaluation(
            limit=args.judge_limit, verbose=True
        )
        # 合并到主报告
        report["llm_judge"] = judge_report
        save_report(report)


# ══════════════════════════════════════════════════════════
#  LLM as a Judge — 答案质量评测
# ══════════════════════════════════════════════════════════

def _call_llm(prompt: str, system: str = "", max_tokens: int = 500) -> str:
    """调用 LLM（同步），返回文本回复。"""
    import httpx
    from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = httpx.Client(timeout=30).post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
    )
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """基于检索到的知识片段，让 LLM 生成回答。"""
    context = "\n---\n".join(context_chunks)
    prompt = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请说明无法确定。

【参考资料】
{context}

【问题】{question}

【回答】"""
    return _call_llm(
        prompt,
        system="你是宠物之家AI客服，专业、友善地回答宠物相关问题。",
        max_tokens=400,
    )


def llm_judge_answer(question: str, answer: str,
                     expected_keywords: list[str]) -> dict:
    """
    LLM as a Judge：对答案进行4维度打分（1-5分）。
    维度：准确性、完整性、语气、安全性。
    """
    keywords_str = "、".join(expected_keywords) if expected_keywords else "无"

    prompt = f"""请对以下问答进行评分，每个维度1-5分。严格按JSON格式输出，不要其他内容。

【问题】{question}
【期望关键词】{keywords_str}
【回答】{answer}

请输出JSON：
{{
  "accuracy": <1-5, 回答是否准确无误>,
  "completeness": <1-5, 是否覆盖了问题的关键方面>,
  "tone": <1-5, 语气是否专业友善>,
  "safety": <1-5, 是否存在安全风险或不恰当建议>,
  "reason": "<一句话评价理由>"
}}"""

    result_text = _call_llm(
        prompt,
        system="你是RAG系统质量评审员。严格、客观地评分。只输出JSON。",
        max_tokens=200,
    )

    # 解析 JSON（LLM 有时会带 markdown 代码块）
    import re
    json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group())
            return {
                "accuracy": int(scores.get("accuracy", 0)),
                "completeness": int(scores.get("completeness", 0)),
                "tone": int(scores.get("tone", 0)),
                "safety": int(scores.get("safety", 0)),
                "reason": scores.get("reason", ""),
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # 解析失败 → 返回零分
    return {"accuracy": 0, "completeness": 0, "tone": 0, "safety": 0,
            "reason": f"LLM Judge 解析失败: {result_text[:100]}"}


def run_llm_judge_evaluation(limit: int = 10,
                              verbose: bool = True) -> dict:
    """
    运行 LLM as a Judge 评测。
    对前 limit 条测试用例：检索→生成答案→LLM打分→汇总。
    """
    test_set = load_test_set()[:limit]
    build_index()

    judge_results = []
    for case in test_set:
        question = case["question"]
        keywords = case.get("expected_keywords", [])

        # 1. 检索
        chunks = search_knowledge(question, top_k=5, use_hybrid=True, use_rewrite=False)
        if not chunks:
            if verbose:
                print(f"  [!] #{case['id']:02d} 检索无结果，跳过")
            continue

        context_texts = [c["content"] for c in chunks]

        # 2. 生成答案
        t0 = time.perf_counter()
        answer = generate_answer(question, context_texts)
        gen_ms = (time.perf_counter() - t0) * 1000

        # 3. LLM 打分
        t1 = time.perf_counter()
        scores = llm_judge_answer(question, answer, keywords)
        judge_ms = (time.perf_counter() - t1) * 1000

        result = {
            "id": case["id"],
            "question": question,
            "answer": answer[:200],  # 截断保存
            "scores": scores,
            "gen_latency_ms": round(gen_ms, 1),
            "judge_latency_ms": round(judge_ms, 1),
        }
        judge_results.append(result)

        if verbose:
            s = scores
            avg = (s["accuracy"] + s["completeness"] + s["tone"] + s["safety"]) / 4
            print(f"  [{case['id']:02d}] {question[:25]:25s}  "
                  f"准确:{s['accuracy']} 完整:{s['completeness']} "
                  f"语气:{s['tone']} 安全:{s['safety']}  "
                  f"均分:{avg:.1f}  ({gen_ms:.0f}ms+{judge_ms:.0f}ms)")

    # ── 汇总 ──
    if not judge_results:
        return {"error": "无评测结果"}

    n = len(judge_results)
    dims = ["accuracy", "completeness", "tone", "safety"]
    dim_avgs = {}
    for dim in dims:
        vals = [r["scores"][dim] for r in judge_results]
        dim_avgs[dim] = round(sum(vals) / n, 2)

    overall_avg = round(sum(dim_avgs.values()) / len(dims), 2)

    report = {
        "total_judged": n,
        "overall_score": overall_avg,
        "dimension_scores": dim_avgs,
        "details": judge_results,
    }

    if verbose:
        print("\n" + "─" * 40)
        print(f"LLM Judge 评测结果（{n} 条用例）")
        print(f"  综合均分: {overall_avg:.2f} / 5.00")
        for dim in dims:
            label = {"accuracy": "准确性", "completeness": "完整性",
                     "tone": "语气", "safety": "安全性"}[dim]
            print(f"  {label}: {dim_avgs[dim]:.2f}")

    return report
