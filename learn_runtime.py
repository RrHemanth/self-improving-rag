import json

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker
from src.agent.query_rewriter import QueryRewriter
from src.agent.self_improving_agent import (
    SelfImprovingRAGAgent
)
from src.generation.generator import Generator
from src.evaluation.rag_evaluator import RAGEvaluator
from src.memory.experience_memory import (
    ExperienceMemory
)


# --------------------------------------------------
# Runtime learning settings
# --------------------------------------------------

BATCH_SIZE = 3

RUNTIME_QUERY_FILE = (
    "data/training/runtime_queries.json"
)

MEMORY_FILE = (
    "data/memory/experiences.json"
)


# --------------------------------------------------
# Load document
# --------------------------------------------------

print("Loading document...")

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

print("\nLoading models...")

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts
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


# --------------------------------------------------
# Build RAG components
# --------------------------------------------------

rewriter = QueryRewriter()

generator = Generator()

# Evaluator is now frozen.
evaluator = RAGEvaluator()

memory = ExperienceMemory(
    MEMORY_FILE
)


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
# Load runtime queries
# --------------------------------------------------

with open(
    RUNTIME_QUERY_FILE
) as file:
    runtime_queries = json.load(file)


total_queries = len(
    runtime_queries
)


# --------------------------------------------------
# Find queries already learned
# --------------------------------------------------

existing_queries = {
    experience["query"]
    for experience in memory.experiences
}


unseen_queries = [
    query
    for query in runtime_queries
    if query not in existing_queries
]


already_learned = (
    total_queries
    - len(unseen_queries)
)


print("\n" + "=" * 80)
print("RUNTIME LEARNING")
print("=" * 80)

print(
    "\nTotal runtime queries:",
    total_queries
)

print(
    "Already learned:",
    already_learned
)

print(
    "Remaining:",
    len(unseen_queries)
)


# --------------------------------------------------
# Stop if everything is complete
# --------------------------------------------------

if not unseen_queries:
    print(
        "\nAll runtime queries have already "
        "been learned."
    )

    print(
        "\nExperience memory:"
    )

    print(
        MEMORY_FILE
    )

    raise SystemExit


# --------------------------------------------------
# Take only the next batch
# --------------------------------------------------

batch = unseen_queries[
    :BATCH_SIZE
]


print(
    "\nProcessing this run:",
    len(batch),
    "queries"
)


# --------------------------------------------------
# Batch statistics
# --------------------------------------------------

strategy_wins = {
    strategy: 0
    for strategy in agent.strategies
}


tie_count = 0
processed = 0


# --------------------------------------------------
# Runtime learning
# --------------------------------------------------

for batch_index, query in enumerate(
    batch,
    1
):
    global_index = (
        already_learned
        + batch_index
    )

    print("\n" + "=" * 80)

    print(
        f"Runtime query "
        f"{global_index}/{total_queries}"
    )

    print(query)


    # During learning we deliberately explore every
    # strategy so memory gets comparative evidence.
    result = agent.run(
        query,
        learn=True,
        force_explore=True
    )

    processed += 1


    print("\nRewritten query:")
    print(
        result["rewritten_query"]
    )


    print("\nStrategy results:")


    for attempt in result["attempts"]:
        print(
            "\n",
            attempt["strategy"],
            sep=""
        )

        print(
            "  Retrieval:",
            round(
                attempt[
                    "retrieval_relevance"
                ],
                3
            )
        )

        print(
            "  Relevant fraction:",
            round(
                attempt[
                    "relevant_fraction"
                ],
                3
            )
        )

        print(
            "  Direct fraction:",
            round(
                attempt[
                    "direct_fraction"
                ],
                3
            )
        )

        print(
            "  Answer relevance:",
            round(
                attempt[
                    "answer_relevance"
                ],
                3
            )
        )

        print(
            "  Faithfulness:",
            round(
                attempt[
                    "faithfulness"
                ],
                3
            )
        )

        print(
            "  Overall:",
            round(
                attempt[
                    "overall_score"
                ],
                3
            )
        )

        print(
            "  Diagnosis:",
            attempt["diagnosis"]
        )

        print(
            "  Success:",
            attempt["success"]
        )


    # --------------------------------------------------
    # Best strategy / tie
    # --------------------------------------------------

    print("\nBest result:")


    if result["best_strategy_tie"]:
        tie_count += 1

        print(
            "  Tie between:",
            ", ".join(
                result[
                    "best_strategies"
                ]
            )
        )

    else:
        print(
            "  Strategy:",
            result[
                "best_strategy"
            ]
        )


    print(
        "  Score:",
        round(
            result[
                "best_score"
            ],
            3
        )
    )


    for strategy in result[
        "best_strategies"
    ]:
        strategy_wins[
            strategy
        ] += 1


    print("\nFinal answer:")

    print(
        result["answer"]
    )


# --------------------------------------------------
# Updated progress
# --------------------------------------------------

learned_after_run = len(
    memory.experiences
)

remaining_after_run = (
    total_queries
    - learned_after_run
)


print("\n")
print("=" * 80)
print("RUNTIME LEARNING BATCH COMPLETE")
print("=" * 80)


print(
    "\nQueries processed this run:",
    processed
)

print(
    "Total experiences stored:",
    learned_after_run
)

print(
    "Queries remaining:",
    max(
        remaining_after_run,
        0
    )
)


print(
    "\nQueries with a top-score tie "
    "this run:",
    tie_count
)


print(
    "\nBest / tied-best strategies "
    "this run:"
)


for strategy, count in strategy_wins.items():
    print(
        strategy,
        ":",
        count
    )


print(
    "\nExperience memory:"
)

print(
    MEMORY_FILE
)


if remaining_after_run > 0:
    print(
        "\nRun python3 learn_runtime.py "
        "again later to process the next batch."
    )

else:
    print(
        "\nRuntime learning is complete."
    )