class HybridRetriever:
    def __init__(self, faiss_store, bm25_store, embedding_model):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.embedding_model = embedding_model

    def search(self, query, top_k=5, candidate_k=10):
        query_embedding = self.embedding_model.encode([query])[0]

        faiss_results = self.faiss_store.search(
            query_embedding,
            top_k=candidate_k
        )

        bm25_results = self.bm25_store.search(
            query,
            top_k=candidate_k
        )

        scores = {}
        chunks = {}

        for rank, chunk in enumerate(faiss_results, 1):
            key = (
                chunk["source"],
                chunk["page"],
                chunk["chunk_id"]
            )

            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            chunks[key] = chunk

        for rank, chunk in enumerate(bm25_results, 1):
            key = (
                chunk["source"],
                chunk["page"],
                chunk["chunk_id"]
            )

            scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            chunks[key] = chunk

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for key, score in ranked[:top_k]:
            chunk = chunks[key].copy()
            chunk["hybrid_score"] = score
            results.append(chunk)

        return results