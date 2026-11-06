import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker
from src.agent.query_rewriter import QueryRewriter
from src.agent.self_improving_agent import SelfImprovingRAGAgent
from src.generation.generator import Generator
from src.evaluation.rag_evaluator import RAGEvaluator
from src.memory.experience_memory import ExperienceMemory


# --------------------------------------------------
# Final benchmark settings
# --------------------------------------------------

BATCH_SIZE = 6

MEMORY_FILE = Path(
    "data/memory/experiences_frozen.json"
)

QUESTION_FILE = Path(
    "data/evaluation/questions.json"
)

RUNTIME_QUERY_FILE = Path(
    "data/training/runtime_queries.json"
)

RESULTS_FILE = Path(
    "data/evaluation/"
    "system_ablation_full_results.json"
)

SUMMARY_FILE = Path(
    "data/evaluation/"
    "system_ablation_full_summary.json"
)


SYSTEM_NAMES = [
    "baseline_dense",
    "static_advanced",
    "explore_all_agent",
    "self_improving"
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_json_atomic(
    path,
    data
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_path = Path(
        str(path) + ".tmp"
    )

    with open(
        temp_path,
        "w"
    ) as file:
        json.dump(
            data,
            file,
            indent=2
        )

    temp_path.replace(
        path
    )


def average(values):
    if not values:
        return 0.0

    return (
        sum(values)
        / len(values)
    )


# --------------------------------------------------
# Retrieval metrics
# --------------------------------------------------

def hit_at_k(
    results,
    relevant_pages,
    k=5
):
    retrieved = [
        result["page"]
        for result in results[:k]
    ]

    return int(
        any(
            page in relevant_pages
            for page in retrieved
        )
    )


def precision_at_k(
    results,
    relevant_pages,
    k=5
):
    retrieved = [
        result["page"]
        for result in results[:k]
    ]

    if not retrieved:
        return 0.0

    relevant_pages = set(
        relevant_pages
    )

    relevant_count = sum(
        page in relevant_pages
        for page in retrieved
    )

    return (
        relevant_count
        / len(retrieved)
    )


def recall_at_k(
    results,
    relevant_pages,
    k=5
):
    retrieved = set(
        result["page"]
        for result in results[:k]
    )

    relevant_pages = set(
        relevant_pages
    )

    if not relevant_pages:
        return 0.0

    return (
        len(
            retrieved.intersection(
                relevant_pages
            )
        )
        / len(relevant_pages)
    )


def reciprocal_rank(
    results,
    relevant_pages
):
    relevant_pages = set(
        relevant_pages
    )

    for rank, result in enumerate(
        results,
        1
    ):
        if (
            result["page"]
            in relevant_pages
        ):
            return 1 / rank

    return 0.0


def retrieval_metrics(
    results,
    relevant_pages
):
    return {
        "hit": hit_at_k(
            results,
            relevant_pages
        ),

        "precision": precision_at_k(
            results,
            relevant_pages
        ),

        "recall": recall_at_k(
            results,
            relevant_pages
        ),

        "rr": reciprocal_rank(
            results,
            relevant_pages
        )
    }


# --------------------------------------------------
# Answer evaluation
# --------------------------------------------------

def evaluate_generated_answer(
    evaluator,
    query,
    results,
    answer
):
    if not answer:
        return {
            "answer_relevance": 0.0,
            "faithfulness": 0.0
        }

    evaluation = (
        evaluator.evaluate_answer(
            query,
            results,
            answer
        )
    )

    return {
        "answer_relevance":
            evaluation[
                "answer_relevance"
            ],

        "faithfulness":
            evaluation[
                "faithfulness"
            ]
    }


# --------------------------------------------------
# Convert agent result into benchmark record
# --------------------------------------------------

def build_agent_record(
    result,
    relevant_pages,
    latency
):
    retrieved_results = (
        result["results"]
        or []
    )

    metrics = retrieval_metrics(
        retrieved_results,
        relevant_pages
    )

    best_attempt = max(
        result["attempts"],
        key=lambda attempt:
            attempt["overall_score"]
    )

    return {
        **metrics,

        "answer_relevance":
            best_attempt[
                "answer_relevance"
            ],

        "faithfulness":
            best_attempt[
                "faithfulness"
            ],

        "latency":
            latency,

        "attempts":
            len(
                result["attempts"]
            ),

        "mode":
            result["mode"],

        "initial_strategy":
            result[
                "initial_strategy"
            ],

        "strategy":
            result[
                "best_strategy"
            ],

        "best_strategies":
            result.get(
                "best_strategies",
                [
                    result[
                        "best_strategy"
                    ]
                ]
            ),

        "best_strategy_tie":
            result.get(
                "best_strategy_tie",
                False
            )
    }


# --------------------------------------------------
# Verify frozen memory
# --------------------------------------------------

if not MEMORY_FILE.exists():
    raise FileNotFoundError(
        f"Frozen memory not found: "
        f"{MEMORY_FILE}"
    )


memory_hash_before = hashlib.sha256(
    MEMORY_FILE.read_bytes()
).hexdigest()


with open(
    MEMORY_FILE
) as file:
    frozen_experiences = json.load(
        file
    )


if len(frozen_experiences) != 40:
    raise ValueError(
        "Expected exactly 40 frozen "
        "experiences, found "
        f"{len(frozen_experiences)}."
    )


frozen_queries = {
    experience["query"]
    for experience
    in frozen_experiences
}


if len(frozen_queries) != 40:
    raise ValueError(
        "Frozen memory contains "
        "duplicate queries."
    )


print(
    "Frozen runtime experiences:",
    len(frozen_experiences)
)


# --------------------------------------------------
# Load held-out benchmark
# --------------------------------------------------

with open(
    QUESTION_FILE
) as file:
    questions = json.load(
        file
    )


if len(questions) != 64:
    raise ValueError(
        "Expected 64 held-out questions, "
        f"found {len(questions)}."
    )


benchmark_queries = {
    item["query"].strip().lower()
    for item in questions
}


if len(benchmark_queries) != 64:
    raise ValueError(
        "Benchmark contains duplicate "
        "questions."
    )


print(
    "Held-out benchmark questions:",
    len(questions)
)


# --------------------------------------------------
# Verify runtime / benchmark separation
# --------------------------------------------------

with open(
    RUNTIME_QUERY_FILE
) as file:
    runtime_data = json.load(
        file
    )


runtime_queries = []

for item in runtime_data:
    if isinstance(item, str):
        runtime_queries.append(
            item
        )

    elif isinstance(item, dict):
        runtime_queries.append(
            item.get(
                "query",
                ""
            )
        )


runtime_set = {
    query.strip().lower()
    for query in runtime_queries
    if query.strip()
}


overlap = (
    runtime_set
    & benchmark_queries
)


if overlap:
    raise ValueError(
        "Runtime learning set overlaps "
        "with held-out benchmark."
    )


print(
    "Exact train / benchmark overlap:",
    len(overlap)
)


# --------------------------------------------------
# Load existing benchmark progress
# --------------------------------------------------

if RESULTS_FILE.exists():
    with open(
        RESULTS_FILE
    ) as file:
        completed_results = (
            json.load(file)
        )

else:
    completed_results = []


completed_queries = {
    result["query"]
    for result in completed_results
}


if (
    len(completed_queries)
    != len(completed_results)
):
    raise ValueError(
        "Duplicate benchmark results "
        "detected."
    )


valid_benchmark_queries = {
    item["query"]
    for item in questions
}


for result in completed_results:
    if (
        result["query"]
        not in valid_benchmark_queries
    ):
        raise ValueError(
            "Existing results contain "
            "an unknown benchmark query."
        )

    for system_name in SYSTEM_NAMES:
        if (
            system_name
            not in result
        ):
            raise ValueError(
                "Existing benchmark result "
                "uses an incompatible format."
            )


remaining_questions = [
    (
        index,
        item
    )
    for index, item in enumerate(
        questions,
        1
    )
    if (
        item["query"]
        not in completed_queries
    )
]


print(
    "Already completed:",
    len(completed_results)
)


print(
    "Remaining:",
    len(remaining_questions)
)


# --------------------------------------------------
# Stop if benchmark already complete
# --------------------------------------------------

if not remaining_questions:
    print(
        "\nAll 64 benchmark questions "
        "are already complete."
    )

else:
    batch = remaining_questions[
        :BATCH_SIZE
    ]

    print(
        "Processing this run:",
        len(batch)
    )


    # --------------------------------------------------
    # Load document
    # --------------------------------------------------

    print(
        "\nLoading document..."
    )


    documents = load_document(
        "data/raw/python_dummies.pdf"
    )


    chunks = chunk_documents(
        documents,
        chunk_size=800,
        overlap=150
    )


    print(
        "Created",
        len(chunks),
        "chunks"
    )


    # --------------------------------------------------
    # Build retrieval components
    # --------------------------------------------------

    print(
        "\nLoading models..."
    )


    embedding_model = EmbeddingModel()


    texts = [
        chunk["text"]
        for chunk in chunks
    ]


    embeddings = (
        embedding_model.encode(
            texts
        )
    )


    faiss_store = FaissStore(
        embeddings.shape[1]
    )


    faiss_store.add(
        embeddings,
        chunks
    )


    bm25_store = BM25Store(
        chunks
    )


    hybrid_store = HybridRetriever(
        faiss_store,
        bm25_store,
        embedding_model
    )


    reranker = Reranker()

    rewriter = QueryRewriter()

    generator = Generator()

    evaluator = RAGEvaluator()


    # --------------------------------------------------
    # Load frozen experience memory
    # --------------------------------------------------

    memory = ExperienceMemory(
        str(MEMORY_FILE)
    )


    if len(memory.experiences) != 40:
        raise ValueError(
            "ExperienceMemory did not "
            "load 40 experiences."
        )


    # --------------------------------------------------
    # Build agent
    # --------------------------------------------------

    agent = SelfImprovingRAGAgent(
        faiss_store,
        bm25_store,
        hybrid_store,
        reranker,
        embedding_model,
        rewriter,
        generator,
        evaluator,
        memory
    )


    # --------------------------------------------------
    # Run this batch
    # --------------------------------------------------

    for question_number, item in batch:
        query = item[
            "query"
        ]

        relevant_pages = item[
            "relevant_pages"
        ]


        print(
            "\n"
            + "=" * 80
        )


        print(
            f"Question "
            f"{question_number}/64"
        )


        print(query)


        # ==============================================
        # 1. Baseline dense RAG
        # ==============================================

        print(
            "\nRunning baseline dense..."
        )


        start = (
            time.perf_counter()
        )


        query_embedding = (
            embedding_model.encode(
                [query]
            )[0]
        )


        baseline_results = (
            faiss_store.search(
                query_embedding,
                top_k=5
            )
        )


        baseline_answer = (
            generator.generate(
                query,
                baseline_results
            )
        )


        baseline_answer_eval = (
            evaluate_generated_answer(
                evaluator,
                query,
                baseline_results,
                baseline_answer
            )
        )


        baseline_time = (
            time.perf_counter()
            - start
        )


        baseline_metrics = (
            retrieval_metrics(
                baseline_results,
                relevant_pages
            )
        )


        baseline_record = {
            **baseline_metrics,
            **baseline_answer_eval,

            "latency":
                baseline_time,

            "attempts":
                1
        }


        # ==============================================
        # 2. Static advanced RAG
        # ==============================================

        print(
            "Running static advanced..."
        )


        start = (
            time.perf_counter()
        )


        rewritten_query = (
            rewriter.rewrite(
                query
            )
        )


        candidates = (
            hybrid_store.search(
                rewritten_query,
                top_k=10,
                candidate_k=15
            )
        )


        static_results = (
            reranker.rerank(
                query,
                candidates,
                top_k=5
            )
        )


        static_answer = (
            generator.generate(
                query,
                static_results
            )
        )


        static_answer_eval = (
            evaluate_generated_answer(
                evaluator,
                query,
                static_results,
                static_answer
            )
        )


        static_time = (
            time.perf_counter()
            - start
        )


        static_metrics = (
            retrieval_metrics(
                static_results,
                relevant_pages
            )
        )


        static_record = {
            **static_metrics,
            **static_answer_eval,

            "latency":
                static_time,

            "attempts":
                1
        }


        # ==============================================
        # 3. Explore-all adaptive agent
        #
        # No memory shortcut.
        # Always tests all four strategies.
        # ==============================================

        print(
            "Running explore-all agent..."
        )


        start = (
            time.perf_counter()
        )


        explore_all_result = (
            agent.run(
                query,
                learn=False,
                force_explore=True
            )
        )


        explore_all_time = (
            time.perf_counter()
            - start
        )


        explore_all_record = (
            build_agent_record(
                explore_all_result,
                relevant_pages,
                explore_all_time
            )
        )


        # ==============================================
        # 4. Self-improving agent
        #
        # Uses frozen 40-experience memory.
        # ==============================================

        print(
            "Running self-improving..."
        )


        start = (
            time.perf_counter()
        )


        adaptive_result = (
            agent.run(
                query,
                learn=False
            )
        )


        adaptive_time = (
            time.perf_counter()
            - start
        )


        adaptive_record = (
            build_agent_record(
                adaptive_result,
                relevant_pages,
                adaptive_time
            )
        )


        # ==============================================
        # Save complete question result
        # ==============================================

        question_record = {
            "question_number":
                question_number,

            "query":
                query,

            "relevant_pages":
                relevant_pages,

            "query_type":
                item.get(
                    "query_type"
                ),

            "difficulty":
                item.get(
                    "difficulty"
                ),

            "baseline_dense":
                baseline_record,

            "static_advanced":
                static_record,

            "explore_all_agent":
                explore_all_record,

            "self_improving":
                adaptive_record
        }


        completed_results.append(
            question_record
        )


        completed_results.sort(
            key=lambda result:
                result[
                    "question_number"
                ]
        )


        save_json_atomic(
            RESULTS_FILE,
            completed_results
        )


        # ==============================================
        # Question summary
        # ==============================================

        print(
            "\nBaseline MRR:",
            round(
                baseline_record[
                    "rr"
                ],
                3
            )
        )


        print(
            "Static MRR:",
            round(
                static_record[
                    "rr"
                ],
                3
            )
        )


        print(
            "Explore-all MRR:",
            round(
                explore_all_record[
                    "rr"
                ],
                3
            )
        )


        print(
            "Self-improving MRR:",
            round(
                adaptive_record[
                    "rr"
                ],
                3
            )
        )


        print(
            "Explore-all attempts:",
            explore_all_record[
                "attempts"
            ]
        )


        print(
            "Self-improving mode:",
            adaptive_record[
                "mode"
            ]
        )


        print(
            "Self-improving initial strategy:",
            adaptive_record[
                "initial_strategy"
            ]
        )


        print(
            "Self-improving final strategy:",
            adaptive_record[
                "strategy"
            ]
        )


        print(
            "Self-improving attempts:",
            adaptive_record[
                "attempts"
            ]
        )


        print(
            "\nSaved progress:",
            f"{len(completed_results)}/64"
        )


# --------------------------------------------------
# Verify frozen memory file was not modified
# --------------------------------------------------

memory_hash_after = hashlib.sha256(
    MEMORY_FILE.read_bytes()
).hexdigest()


if (
    memory_hash_before
    != memory_hash_after
):
    raise RuntimeError(
        "Frozen experience memory "
        "was modified."
    )


print(
    "\nFrozen memory unchanged."
)


# --------------------------------------------------
# Current progress
# --------------------------------------------------

with open(
    RESULTS_FILE
) as file:
    completed_results = (
        json.load(file)
    )


completed_count = len(
    completed_results
)


remaining_count = (
    64
    - completed_count
)


print(
    "\n"
    + "=" * 80
)


print(
    "FINAL BENCHMARK PROGRESS"
)


print(
    "=" * 80
)


print(
    "\nCompleted:",
    completed_count,
    "/ 64"
)


print(
    "Remaining:",
    remaining_count
)


# --------------------------------------------------
# Final aggregation
# --------------------------------------------------

if completed_count == 64:
    print(
        "\nAll benchmark questions "
        "are complete."
    )


    systems = {
        system_name: [
            result[
                system_name
            ]
            for result
            in completed_results
        ]
        for system_name
        in SYSTEM_NAMES
    }


    summary = {}


    for (
        system_name,
        records
    ) in systems.items():
        summary[
            system_name
        ] = {
            "hit_at_5":
                average([
                    record[
                        "hit"
                    ]
                    for record
                    in records
                ]),

            "precision_at_5":
                average([
                    record[
                        "precision"
                    ]
                    for record
                    in records
                ]),

            "recall_at_5":
                average([
                    record[
                        "recall"
                    ]
                    for record
                    in records
                ]),

            "mrr":
                average([
                    record[
                        "rr"
                    ]
                    for record
                    in records
                ]),

            "answer_relevance":
                average([
                    record[
                        "answer_relevance"
                    ]
                    for record
                    in records
                ]),

            "faithfulness":
                average([
                    record[
                        "faithfulness"
                    ]
                    for record
                    in records
                ]),

            "average_latency":
                average([
                    record[
                        "latency"
                    ]
                    for record
                    in records
                ]),

            "average_attempts":
                average([
                    record[
                        "attempts"
                    ]
                    for record
                    in records
                ])
        }


    # --------------------------------------------------
    # MRR improvement versus dense baseline
    # --------------------------------------------------

    baseline_mrr = (
        summary[
            "baseline_dense"
        ]["mrr"]
    )


    for name in [
        "static_advanced",
        "explore_all_agent",
        "self_improving"
    ]:
        current_mrr = (
            summary[
                name
            ]["mrr"]
        )

        if baseline_mrr > 0:
            improvement = (
                (
                    current_mrr
                    - baseline_mrr
                )
                / baseline_mrr
                * 100
            )

        else:
            improvement = 0.0


        summary[
            name
        ][
            "mrr_improvement_vs_baseline_pct"
        ] = improvement


    # --------------------------------------------------
    # Self-improving efficiency
    # --------------------------------------------------

    explore_attempts = (
        summary[
            "explore_all_agent"
        ][
            "average_attempts"
        ]
    )


    adaptive_attempts = (
        summary[
            "self_improving"
        ][
            "average_attempts"
        ]
    )


    if explore_attempts > 0:
        attempt_reduction = (
            (
                explore_attempts
                - adaptive_attempts
            )
            / explore_attempts
            * 100
        )

    else:
        attempt_reduction = 0.0


    summary[
        "self_improving"
    ][
        "attempt_reduction_vs_explore_all_pct"
    ] = attempt_reduction


    explore_latency = (
        summary[
            "explore_all_agent"
        ][
            "average_latency"
        ]
    )


    adaptive_latency = (
        summary[
            "self_improving"
        ][
            "average_latency"
        ]
    )


    if explore_latency > 0:
        latency_reduction = (
            (
                explore_latency
                - adaptive_latency
            )
            / explore_latency
            * 100
        )

    else:
        latency_reduction = 0.0


    summary[
        "self_improving"
    ][
        "latency_reduction_vs_explore_all_pct"
    ] = latency_reduction


    # --------------------------------------------------
    # Agent diagnostics
    # --------------------------------------------------

    mode_counts = Counter(
        record[
            "self_improving"
        ][
            "mode"
        ]
        for record
        in completed_results
    )


    initial_strategy_counts = Counter(
        record[
            "self_improving"
        ][
            "initial_strategy"
        ]
        for record
        in completed_results
    )


    final_strategy_counts = Counter(
        record[
            "self_improving"
        ][
            "strategy"
        ]
        for record
        in completed_results
    )


    explore_strategy_counts = Counter(
        record[
            "explore_all_agent"
        ][
            "strategy"
        ]
        for record
        in completed_results
    )


    summary[
        "self_improving"
    ][
        "mode_counts"
    ] = dict(
        mode_counts
    )


    summary[
        "self_improving"
    ][
        "initial_strategy_counts"
    ] = dict(
        initial_strategy_counts
    )


    summary[
        "self_improving"
    ][
        "final_strategy_counts"
    ] = dict(
        final_strategy_counts
    )


    summary[
        "explore_all_agent"
    ][
        "final_strategy_counts"
    ] = dict(
        explore_strategy_counts
    )


    # --------------------------------------------------
    # Save final summary
    # --------------------------------------------------

    save_json_atomic(
        SUMMARY_FILE,
        summary
    )


    # --------------------------------------------------
    # Print final results
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )


    print(
        "FINAL 64-QUESTION BENCHMARK"
    )


    print(
        "=" * 80
    )


    for (
        name,
        metrics
    ) in summary.items():
        print(
            "\n",
            name.upper()
        )


        print(
            "Hit@5:",
            round(
                metrics[
                    "hit_at_5"
                ],
                3
            )
        )


        print(
            "Precision@5:",
            round(
                metrics[
                    "precision_at_5"
                ],
                3
            )
        )


        print(
            "Recall@5:",
            round(
                metrics[
                    "recall_at_5"
                ],
                3
            )
        )


        print(
            "MRR:",
            round(
                metrics[
                    "mrr"
                ],
                3
            )
        )


        print(
            "Answer relevance:",
            round(
                metrics[
                    "answer_relevance"
                ],
                3
            )
        )


        print(
            "Faithfulness:",
            round(
                metrics[
                    "faithfulness"
                ],
                3
            )
        )


        print(
            "Average latency:",
            round(
                metrics[
                    "average_latency"
                ],
                2
            ),
            "seconds"
        )


        print(
            "Average attempts:",
            round(
                metrics[
                    "average_attempts"
                ],
                2
            )
        )


        if (
            "mrr_improvement_vs_baseline_pct"
            in metrics
        ):
            print(
                "MRR improvement vs baseline:",
                round(
                    metrics[
                        "mrr_improvement_vs_baseline_pct"
                    ],
                    2
                ),
                "%"
            )


        if (
            "attempt_reduction_vs_explore_all_pct"
            in metrics
        ):
            print(
                "Attempt reduction vs explore-all:",
                round(
                    metrics[
                        "attempt_reduction_vs_explore_all_pct"
                    ],
                    2
                ),
                "%"
            )


        if (
            "latency_reduction_vs_explore_all_pct"
            in metrics
        ):
            print(
                "Latency reduction vs explore-all:",
                round(
                    metrics[
                        "latency_reduction_vs_explore_all_pct"
                    ],
                    2
                ),
                "%"
            )


    print(
        "\n"
        + "=" * 80
    )


    print(
        "SELF-IMPROVING AGENT DIAGNOSTICS"
    )


    print(
        "=" * 80
    )


    print(
        "\nModes:",
        dict(
            mode_counts
        )
    )


    print(
        "\nInitial strategies:",
        dict(
            initial_strategy_counts
        )
    )


    print(
        "\nFinal strategies:",
        dict(
            final_strategy_counts
        )
    )


    print(
        "\nExplore-all final strategies:",
        dict(
            explore_strategy_counts
        )
    )


    print(
        "\nFinal results saved to:"
    )


    print(
        RESULTS_FILE
    )


    print(
        SUMMARY_FILE
    )


else:
    print(
        "\nThis batch is complete."
    )


    print(
        "Let the system cool if needed, "
        "then run:"
    )


    print(
        "\npython3 benchmark_systems.py"
    )


    print(
        "\nIt will resume automatically "
        "from the next unfinished question."
    )