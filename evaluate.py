import json

from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker
from src.evaluation.evaluator import (
    hit_rate,
    recall_at_k,
    reciprocal_rank,
    precision_at_k
)
from src.memory.experience_store import ExperienceStore


documents = load_document("data/raw/python_dummies.pdf")

chunks = chunk_documents(
    documents,
    chunk_size=800,
    overlap=150
)

embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]
embeddings = embedding_model.encode(texts)

faiss_store = FaissStore(embeddings.shape[1])
faiss_store.add(embeddings, chunks)

bm25_store = BM25Store(chunks)

hybrid_store = HybridRetriever(
    faiss_store,
    bm25_store,
    embedding_model
)

reranker = Reranker()

experience_store = ExperienceStore(
    "data/evaluation/experiences.json"
)


with open("data/evaluation/questions.json") as file:
    questions = json.load(file)


scores = {
    "faiss": [],
    "bm25": [],
    "hybrid": [],
    "reranked": []
}


for item in questions:
    query = item["query"]
    relevant_pages = item["relevant_pages"]

    query_embedding = embedding_model.encode([query])[0]

    faiss_results = faiss_store.search(
        query_embedding,
        top_k=5
    )

    bm25_results = bm25_store.search(
        query,
        top_k=5
    )

    hybrid_results = hybrid_store.search(
        query,
        top_k=10,
        candidate_k=15
    )

    reranked_results = reranker.rerank(
        query,
        hybrid_results,
        top_k=5
    )

    systems = {
        "faiss": faiss_results,
        "bm25": bm25_results,
        "hybrid": hybrid_results[:5],
        "reranked": reranked_results
    }

    query_scores = {}

    print("\nQuery:", query)

    for name, results in systems.items():
        hit = hit_rate(results, relevant_pages)
        recall = recall_at_k(results, relevant_pages)
        rr = reciprocal_rank(results, relevant_pages)
        precision = precision_at_k(results,relevant_pages)

        scores[name].append({
            "hit": hit,
            "recall": recall,
            "precision": precision,
            "rr": rr
        })

        query_scores[name] = {
            "hit": hit,
            "recall": recall,
            "precision": precision,
            "rr": rr
        }

        print(
            name,
            "| Hit@5:", round(hit, 3),
            "| Precision@5:", round(precision, 3),
            "| Recall@5:", round(recall, 3),
            "| RR:", round(rr, 3)
        )

    best_strategy = max(
        query_scores,
        key=lambda name: (
            query_scores[name]["precision"],
            query_scores[name]["rr"],
            query_scores[name]["recall"],
            query_scores[name]["hit"]
        )
    )

    experience_store.add({
        "query": query,
        "relevant_pages": relevant_pages,
        "strategies": query_scores,
        "best_strategy": best_strategy
    })


experience_store.save()


print("\nFINAL RESULTS")

for name, values in scores.items():
    count = len(values)

    avg_hit = sum(x["hit"] for x in values) / count
    avg_recall = sum(x["recall"] for x in values) / count
    mrr = sum(x["rr"] for x in values) / count

    print(
        name,
        "| Hit@5:", round(avg_hit, 3),
        "| Recall@5:", round(avg_recall, 3),
        "| MRR:", round(mrr, 3)
    )