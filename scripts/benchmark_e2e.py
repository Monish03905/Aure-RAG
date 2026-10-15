"""AURA full RAG-path benchmark.

Usage:
    python -m scripts.benchmark_e2e 100

Reports P50/P70/P100 for the complete AURA.ask() path and stage timings.
STT is intentionally measured separately because the task brief distinguishes
speech-to-text timing from the RAG-path timing.
"""
import json
import os
import statistics
import sys
from datetime import datetime

from backend.rag.pipeline import AuraRAG


def percentile(values, pct):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def load_queries(n):
    path = "data/raw/dev_sample.jsonl"
    if not os.path.exists(path):
        fallback = [
            "Describe how a corporation is treated as a legal entity.",
            "Explain how computers perform tasks associated with human intelligence.",
            "Describe how machine learning systems learn from data.",
            "Explain which methods are used in data science.",
            "Explain how retrieval-augmented generation uses a knowledge base.",
            "సాధారణంగా కార్పొరేషన్ ఎలా పనిచేస్తుంది?",
        ]
        return (fallback * ((n // len(fallback)) + 1))[:n]

    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(queries) >= n:
                break
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = (item.get("query") or "").strip()
            if q:
                queries.append(q)
    if not queries:
        raise RuntimeError("No usable queries found in data/raw/dev_sample.jsonl")
    return queries


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    queries = load_queries(n)
    aura = AuraRAG()

    # Warmup avoids measuring first-load effects.
    aura.ask(queries[0])

    total = []
    retrieval = []
    generation = []
    guardrail = []
    embedding = []
    faiss_ms = []
    rerank = []

    print("=" * 72)
    print("AURA — FULL RAG-PATH BENCHMARK")
    print("=" * 72)
    print(f"Queries: {len(queries)}")
    print("Target: < 200 ms")
    print()

    for i, query in enumerate(queries, 1):
        result = aura.ask(query)
        s = result.get("stages_ms", {})
        total.append(float(result.get("total_ms", 0)))
        retrieval.append(float(s.get("embedding_ms", 0)) + float(s.get("faiss_ms", 0)))
        generation.append(float(s.get("generation_ms", 0)))
        guardrail.append(float(s.get("guardrail_ms", 0)))
        embedding.append(float(s.get("embedding_ms", 0)))
        faiss_ms.append(float(s.get("faiss_ms", 0)))
        rerank.append(float(s.get("rerank_ms", 0)))
        print(f"[{i:03d}/{len(queries):03d}] total={total[-1]:7.2f} ms | grounded={str(result.get('grounded')):<5} | {query[:55]}")

    metrics = {
        "total_ms": total,
        "retrieval_ms": retrieval,
        "generation_ms": generation,
        "guardrail_ms": guardrail,
        "embedding_ms": embedding,
        "faiss_ms": faiss_ms,
        "rerank_ms": rerank,
    }

    print()
    print("=" * 72)
    print("STAGE                 AVG       P50       P70       P100")
    print("-" * 72)
    for name, values in metrics.items():
        print(f"{name:<18} {statistics.mean(values):8.2f} {percentile(values,50):9.2f} {percentile(values,70):9.2f} {max(values):9.2f}")

    p100 = max(total)
    status = "PASS" if p100 < 200 else "FAIL"
    print("-" * 72)
    print(f"Full RAG P50 : {percentile(total,50):.2f} ms")
    print(f"Full RAG P70 : {percentile(total,70):.2f} ms")
    print(f"Full RAG P100: {p100:.2f} ms")
    print(f"Target       : 200.00 ms")
    print(f"Result       : {status}")
    print("=" * 72)

    os.makedirs("data/benchmark", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("data/benchmark/e2e_benchmark_report.txt", "w", encoding="utf-8") as f:
        f.write("AURA FULL RAG BENCHMARK\n")
        f.write(f"Timestamp: {timestamp}\nQueries: {len(queries)}\n")
        for name, values in metrics.items():
            f.write(f"{name}: avg={statistics.mean(values):.3f}, p50={percentile(values,50):.3f}, p70={percentile(values,70):.3f}, p100={max(values):.3f}\n")
        f.write(f"FULL_RAG_STATUS={status}\n")


if __name__ == "__main__":
    main()
