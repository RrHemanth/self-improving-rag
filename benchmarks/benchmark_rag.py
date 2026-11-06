import argparse
import json
import re
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.app.pipeline import build_agent
from src.memory.experience_memory import ExperienceMemory


CASES = [
    {
        "name": "deque_methods",
        "training_query": (
            "Which deque methods add multiple elements, "
            "and which methods add only one element?"
        ),
        "test_query": (
            "In Python's deque, how can I insert several items "
            "at once instead of adding them individually?"
        ),
        "required_patterns": [r"\bextend\b", r"\bextendleft\b"],
    },
    {
        "name": "while_loop",
        "training_query": (
            "What three tasks must you perform when using a while statement?"
        ),
        "test_query": (
            "What steps should I follow to make sure a Python while loop "
            "eventually terminates?"
        ),
        "required_patterns": [
            r"\b(condition|test)\b",
            r"\b(update|modify|change|increment)\b",
        ],
    },
]


def keyword_check(answer, patterns):
    """A basic answer-key check, not a substitute for human evaluation."""
    return all(
        re.search(pattern, answer, re.IGNORECASE)
        for pattern in patterns
    )


def execute(agent, query, label, **kwargs):
    print(f"\n  {label}", flush=True)

    start = perf_counter()
    result = agent.run(query, **kwargs)
    elapsed = perf_counter() - start

    print(
        f"  Attempts: {len(result['attempts'])} | "
        f"Score: {result['best_score']:.3f} | "
        f"Time: {elapsed:.2f}s",
        flush=True,
    )

    return {
        "mode": result["mode"],
        "memory_match": bool(
            result["memory_details"]["has_memory_match"]
        ),
        "attempt_count": len(result["attempts"]),
        "best_strategy": result["best_strategy"],
        "best_score": float(result["best_score"]),
        "latency_seconds": round(elapsed, 2),
        "answer": result["answer"],
        "attempts": [
            {
                "strategy": attempt["strategy"],
                "score": float(attempt["overall_score"]),
                "success": bool(attempt["success"]),
            }
            for attempt in result["attempts"]
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="data/raw/python_dummies.pdf",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results.json",
    )
    args = parser.parse_args()

    if not Path(args.file).is_file():
        parser.error(f"Document not found: {args.file}")

    print("Building RAG pipeline...", flush=True)
    start = perf_counter()

    pipeline = build_agent(
        file_paths=[args.file],
        provider="ollama",
        model=args.model,
    )

    agent = pipeline["agent"]

    print(
        f"Pipeline ready in {perf_counter() - start:.2f}s",
        flush=True,
    )

    records = []

    with TemporaryDirectory() as folder:
        for index, case in enumerate(CASES):
            print(f"\n{'=' * 55}")
            print(f"CASE: {case['name']}")
            print(f"{'=' * 55}", flush=True)

            # Independent memory for each case.
            # Never modifies application experience memory.
            agent.memory = ExperienceMemory(
                path=str(Path(folder) / f"case_{index}.json")
            )

            training = execute(
                agent,
                case["training_query"],
                "Training: full exploration",
                learn=True,
            )

            baseline = execute(
                agent,
                case["test_query"],
                "Baseline: forced exploration",
                learn=False,
                force_explore=True,
            )

            adaptive = execute(
                agent,
                case["test_query"],
                "Adaptive: memory-guided",
                learn=False,
            )

            baseline["keyword_check"] = bool(
                keyword_check(
                    baseline["answer"],
                    case["required_patterns"],
                )
            )

            adaptive["keyword_check"] = bool(
                keyword_check(
                    adaptive["answer"],
                    case["required_patterns"],
                )
            )

            reduction = 100 * (
                1
                - adaptive["attempt_count"]
                / baseline["attempt_count"]
            )

            records.append({
                "case": case["name"],
                "training_query": case["training_query"],
                "test_query": case["test_query"],
                "training": training,
                "baseline": baseline,
                "adaptive": adaptive,
                "attempt_reduction_percent": round(reduction, 2),
                "quality_delta": round(
                    adaptive["best_score"] - baseline["best_score"],
                    4,
                ),
            })

    baseline_attempts = sum(
        r["baseline"]["attempt_count"] for r in records
    )
    adaptive_attempts = sum(
        r["adaptive"]["attempt_count"] for r in records
    )
    baseline_latency = sum(
        r["baseline"]["latency_seconds"] for r in records
    )
    adaptive_latency = sum(
        r["adaptive"]["latency_seconds"] for r in records
    )

    summary = {
        "case_count": len(records),
        "baseline_attempts": baseline_attempts,
        "adaptive_attempts": adaptive_attempts,
        "attempt_reduction_percent": round(
            100 * (1 - adaptive_attempts / baseline_attempts), 2
        ),
        "baseline_latency_seconds": round(baseline_latency, 2),
        "adaptive_latency_seconds": round(adaptive_latency, 2),
        "latency_reduction_percent": round(
            100 * (1 - adaptive_latency / baseline_latency), 2
        ),
        "memory_matches": sum(
            r["adaptive"]["memory_match"] for r in records
        ),
        "baseline_keyword_passes": sum(
            r["baseline"]["keyword_check"] for r in records
        ),
        "adaptive_keyword_passes": sum(
            r["adaptive"]["keyword_check"] for r in records
        ),
    }

    report = {
        "document": args.file,
        "generation_model": args.model,
        "evaluation_model": "qwen2.5:7b",
        "summary": summary,
        "cases": records,
        "limitations": (
            "Small benchmark with one run per case. "
            "Keyword checks are heuristic, and LLM evaluation "
            "scores are not independent ground-truth accuracy. "
            "Latency may vary between runs."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print("\n========== BENCHMARK SUMMARY ==========")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved results: {output}")
    print("=======================================")


if __name__ == "__main__":
    main()
