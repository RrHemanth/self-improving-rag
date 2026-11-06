class AdaptiveRetriever:
    def __init__(
        self,
        faiss_store,
        bm25_store,
        hybrid_store,
        reranker,
        embedding_model,
        strategy_selector,
        relevance_evaluator,
        experience_store
    ):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.hybrid_store = hybrid_store
        self.reranker = reranker
        self.embedding_model = embedding_model
        self.strategy_selector = strategy_selector
        self.relevance_evaluator = relevance_evaluator
        self.experience_store = experience_store

    def run_strategy(self, query, strategy, top_k=5):
        if strategy == "faiss":
            query_embedding = self.embedding_model.encode([query])[0]

            return self.faiss_store.search(
                query_embedding,
                top_k=top_k
            )

        if strategy == "bm25":
            return self.bm25_store.search(
                query,
                top_k=top_k
            )

        if strategy == "hybrid":
            return self.hybrid_store.search(
                query,
                top_k=top_k,
                candidate_k=15
            )

        if strategy == "reranked":
            candidates = self.hybrid_store.search(
                query,
                top_k=10,
                candidate_k=15
            )

            return self.reranker.rerank(
                query,
                candidates,
                top_k=top_k
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def retrieve(self, query, top_k=5, threshold=4.5):
        predicted_strategy, selector_details = (
            self.strategy_selector.select(query)
        )

        results = self.run_strategy(
            query,
            predicted_strategy,
            top_k
        )

        initial_score = self.relevance_evaluator.score(
            query,
            results
        )

        strategy_scores = {
            predicted_strategy: initial_score
        }

        strategy_results = {
            predicted_strategy: results
        }

        if initial_score < threshold:
            strategies = [
                "faiss",
                "bm25",
                "hybrid",
                "reranked"
            ]

            for strategy in strategies:
                if strategy == predicted_strategy:
                    continue

                results = self.run_strategy(
                    query,
                    strategy,
                    top_k
                )

                score = self.relevance_evaluator.score(
                    query,
                    results
                )

                strategy_scores[strategy] = score
                strategy_results[strategy] = results

        best_strategy = max(
            strategy_scores,
            key=strategy_scores.get
        )

        best_results = strategy_results[best_strategy]

        experience = {
            "query": query,
            "predicted_strategy": predicted_strategy,
            "strategy_scores": strategy_scores,
            "best_strategy": best_strategy
        }

        self.experience_store.upsert(experience)
        self.experience_store.save()

        self.strategy_selector.add_experience(experience)

        details = {
            "predicted_strategy": predicted_strategy,
            "best_strategy": best_strategy,
            "strategy_scores": strategy_scores,
            "selector": selector_details,
            "retried": initial_score < threshold
        }

        return best_strategy, best_results, details