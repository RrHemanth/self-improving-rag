from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks

        self.tokenized_chunks = [
            chunk["text"].lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query, top_k=5):
        query_tokens = query.lower().split()

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index].copy()
            chunk["score"] = float(scores[index])

            results.append(chunk)

        return results