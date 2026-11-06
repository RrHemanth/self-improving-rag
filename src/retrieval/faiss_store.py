import faiss
import numpy as np


class FaissStore:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks = []

    def add(self, embeddings, chunks):
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index].copy()
            chunk["score"] = float(score)

            results.append(chunk)

        return results