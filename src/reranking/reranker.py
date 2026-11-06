from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_k=5):
        pairs = [
            [query, chunk["text"]]
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        results = []

        for chunk, score in zip(chunks, scores):
            item = chunk.copy()
            item["rerank_score"] = float(score)
            results.append(item)

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return results[:top_k]