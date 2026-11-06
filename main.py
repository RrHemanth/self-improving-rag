from src.ingestion.loader import load_document
from src.ingestion.chunker import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.faiss_store import FaissStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever
from src.reranking.reranker import Reranker


documents = load_document("data/raw/python_dummies.pdf")

chunks = chunk_documents(
    documents,
    chunk_size=10000,
    overlap=1000
)

print(f"Created {len(chunks)} chunks")


# FAISS setup
embedding_model = EmbeddingModel()

texts = [chunk["text"] for chunk in chunks]

print("Creating embeddings...")
embeddings = embedding_model.encode(texts)

faiss_store = FaissStore(embeddings.shape[1])
faiss_store.add(embeddings, chunks)


# BM25 setup
bm25_store = BM25Store(chunks)


# Hybrid setup
hybrid_store = HybridRetriever(
    faiss_store,
    bm25_store,
    embedding_model
)


# Reranker setup
reranker = Reranker()


query = "How do you create a function in Python?"


# FAISS search
query_embedding = embedding_model.encode([query])[0]

faiss_results = faiss_store.search(
    query_embedding,
    top_k=5
)


# BM25 search
bm25_results = bm25_store.search(
    query,
    top_k=5
)


# Hybrid search
hybrid_results = hybrid_store.search(
    query,
    top_k=10,
    candidate_k=15
)


# Rerank hybrid candidates
reranked_results = reranker.rerank(
    query,
    hybrid_results,
    top_k=5
)


print("\nFAISS RESULTS")

for i, result in enumerate(faiss_results, 1):
    print("\n" + "=" * 60)
    print(f"Result {i}")
    print("Score:", result["score"])
    print("Page:", result["page"])
    print(result["text"][:300])


print("\nBM25 RESULTS")

for i, result in enumerate(bm25_results, 1):
    print("\n" + "=" * 60)
    print(f"Result {i}")
    print("Score:", result["score"])
    print("Page:", result["page"])
    print(result["text"][:300])


print("\nHYBRID RESULTS")

for i, result in enumerate(hybrid_results, 1):
    print("\n" + "=" * 60)
    print(f"Result {i}")
    print("Score:", result["hybrid_score"])
    print("Page:", result["page"])
    print(result["text"][:300])


print("\nRERANKED RESULTS")

for i, result in enumerate(reranked_results, 1):
    print("\n" + "=" * 60)
    print(f"Result {i}")
    print("Score:", result["rerank_score"])
    print("Page:", result["page"])
    print(result["text"][:300])