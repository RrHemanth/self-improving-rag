class SelfImprovingRAGAgent:
    def __init__(
        self,
        faiss_store,
        bm25_store,
        hybrid_store,
        reranker,
        embedding_model,
        rewriter,
        generator,
        evaluator,
        memory
    ):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.hybrid_store = hybrid_store
        self.reranker = reranker
        self.embedding_model = embedding_model
        self.rewriter = rewriter
        self.generator = generator
        self.evaluator = evaluator
        self.memory = memory

        self.strategies = [
            "hybrid_rewritten",
            "hybrid_original",
            "dense_rewritten",
            "bm25_rewritten"
        ]

        self.default_strategy = "hybrid_rewritten"

        # Scores this close are treated as tied.
        self.tie_tolerance = 1e-6

    def retrieve(
        self,
        query,
        rewritten_query,
        strategy,
        top_k=5
    ):
        if strategy == "hybrid_rewritten":
            candidates = self.hybrid_store.search(
                rewritten_query,
                top_k=10,
                candidate_k=15
            )

        elif strategy == "hybrid_original":
            candidates = self.hybrid_store.search(
                query,
                top_k=10,
                candidate_k=15
            )

        elif strategy == "dense_rewritten":
            query_embedding = self.embedding_model.encode(
                [rewritten_query]
            )[0]

            candidates = self.faiss_store.search(
                query_embedding,
                top_k=10
            )

        elif strategy == "bm25_rewritten":
            candidates = self.bm25_store.search(
                rewritten_query,
                top_k=10
            )

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}"
            )

        return self.reranker.rerank(
            query,
            candidates,
            top_k=top_k
        )

    def calculate_score(
        self,
        retrieval_score,
        answer_relevance,
        faithfulness
    ):
        return (
            0.4 * retrieval_score
            + 0.3 * answer_relevance
            + 0.3 * faithfulness
        )

    def run_attempt(
        self,
        query,
        rewritten_query,
        strategy
    ):
        # --------------------------------------------------
        # Retrieve
        # --------------------------------------------------

        results = self.retrieve(
            query,
            rewritten_query,
            strategy
        )

        # --------------------------------------------------
        # Evaluate retrieval
        # --------------------------------------------------

        retrieval_eval = self.evaluator.evaluate_retrieval(
            query,
            results
        )

        retrieval_score = retrieval_eval[
            "retrieval_relevance"
        ]

        # --------------------------------------------------
        # Generate answer
        # --------------------------------------------------

        answer = self.generator.generate(
            query,
            results
        )

        # --------------------------------------------------
        # Evaluate answer
        # --------------------------------------------------

        answer_eval = self.evaluator.evaluate_answer(
            query,
            results,
            answer
        )

        answer_relevance = answer_eval[
            "answer_relevance"
        ]

        faithfulness = answer_eval[
            "faithfulness"
        ]

        regeneration_attempted = False
        regenerated = False

        # --------------------------------------------------
        # Regenerate weak answers once
        # --------------------------------------------------

        if (
            answer_relevance < 0.75
            or faithfulness < 0.8
        ):
            regeneration_attempted = True

            new_answer = self.generator.regenerate(
                query,
                results,
                answer_eval
            )

            new_eval = self.evaluator.evaluate_answer(
                query,
                results,
                new_answer
            )

            old_quality = (
                answer_relevance
                + faithfulness
            )

            new_quality = (
                new_eval["answer_relevance"]
                + new_eval["faithfulness"]
            )

            if new_quality > old_quality:
                answer = new_answer
                answer_eval = new_eval
                regenerated = True

        # --------------------------------------------------
        # Final scores
        # --------------------------------------------------

        answer_relevance = answer_eval[
            "answer_relevance"
        ]

        faithfulness = answer_eval[
            "faithfulness"
        ]

        overall_score = self.calculate_score(
            retrieval_score,
            answer_relevance,
            faithfulness
        )

        success = (
            answer_relevance >= 0.75
            and faithfulness >= 0.8
        )

        # --------------------------------------------------
        # Diagnosis
        # --------------------------------------------------

        if success:
            diagnosis = "success"

        elif (
            retrieval_score < 0.5
            and retrieval_eval["relevant_fraction"] < 0.6
        ):
            diagnosis = "retrieval_failure"

        elif faithfulness < 0.8:
            diagnosis = "grounding_failure"

        elif answer_relevance < 0.75:
            diagnosis = "answer_relevance_failure"

        else:
            diagnosis = "generation_failure"

        # --------------------------------------------------
        # Attempt record
        # --------------------------------------------------

        attempt = {
            "strategy":
                strategy,

            "retrieval_relevance":
                retrieval_score,

            "relevant_fraction":
                retrieval_eval[
                    "relevant_fraction"
                ],

            "direct_fraction":
                retrieval_eval[
                    "direct_fraction"
                ],

            "retrieved_pages": [
                result["page"]
                for result in results
            ],

            "chunk_scores":
                retrieval_eval["chunks"],

            "answer_relevance":
                answer_relevance,

            "faithfulness":
                faithfulness,

            "overall_score":
                overall_score,

            "success":
                success,

            "diagnosis":
                diagnosis,

            "regeneration_attempted":
                regeneration_attempted,

            "regenerated":
                regenerated
        }

        return attempt, answer, results

    def run(
        self,
        query,
        learn=True,
        force_explore=False
    ):
        # --------------------------------------------------
        # Rewrite query
        # --------------------------------------------------

        rewritten_query = self.rewriter.rewrite(
            query
        )

        # --------------------------------------------------
        # Ask memory for a preferred strategy
        # --------------------------------------------------

        initial_strategy, memory_details = (
            self.memory.suggest_strategy(
                query,
                self.embedding_model,
                self.strategies,
                self.default_strategy
            )
        )

        has_memory_match = memory_details[
            "has_memory_match"
        ]

        # --------------------------------------------------
        # Exploration / exploitation
        # --------------------------------------------------

        if force_explore:
            mode = "explore_forced"

            strategy_order = self.strategies.copy()

        elif not has_memory_match:
            mode = "explore"

            strategy_order = self.strategies.copy()

        else:
            mode = "exploit"

            strategy_order = [
                initial_strategy
            ]

            for strategy in self.strategies:
                if strategy not in strategy_order:
                    strategy_order.append(
                        strategy
                    )

        attempts = []

        answers_by_strategy = {}
        results_by_strategy = {}

        # --------------------------------------------------
        # Run strategies
        # --------------------------------------------------

        for strategy in strategy_order:
            attempt, answer, results = (
                self.run_attempt(
                    query,
                    rewritten_query,
                    strategy
                )
            )

            attempts.append(
                attempt
            )

            answers_by_strategy[
                strategy
            ] = answer

            results_by_strategy[
                strategy
            ] = results

            # During normal exploitation, stop once
            # the learned strategy succeeds.
            if (
                mode == "exploit"
                and attempt["success"]
            ):
                break

        # --------------------------------------------------
        # Determine best score and ties
        # --------------------------------------------------

        best_score = max(
            attempt["overall_score"]
            for attempt in attempts
        )

        best_strategies = [
            attempt["strategy"]
            for attempt in attempts
            if abs(
                attempt["overall_score"]
                - best_score
            ) <= self.tie_tolerance
        ]

        has_best_tie = (
            len(best_strategies) > 1
        )

        # We still need one answer to return.
        #
        # When strategies tie, the first tied strategy
        # supplies the displayed answer, but the experience
        # explicitly retains every tied strategy.
        selected_strategy = (
            best_strategies[0]
        )

        best_answer = answers_by_strategy[
            selected_strategy
        ]

        best_results = results_by_strategy[
            selected_strategy
        ]

        # --------------------------------------------------
        # Store runtime experience
        # --------------------------------------------------

        experience = {
            "query":
                query,

            "rewritten_query":
                rewritten_query,

            "mode":
                mode,

            "initial_strategy":
                initial_strategy,

            # Kept for backward compatibility.
            "best_strategy":
                selected_strategy,

            # This is the field that tells us whether
            # several strategies were equally good.
            "best_strategies":
                best_strategies,

            "best_score":
                best_score,

            "best_strategy_tie":
                has_best_tie,

            # All attempt scores are retained so memory
            # can compare strategies without losing ties.
            "attempts":
                attempts
        }

        if learn:
            self.memory.add(
                experience
            )

        return {
            "query":
                query,

            "rewritten_query":
                rewritten_query,

            "mode":
                mode,

            "initial_strategy":
                initial_strategy,

            "best_strategy":
                selected_strategy,

            "best_strategies":
                best_strategies,

            "best_strategy_tie":
                has_best_tie,

            "best_score":
                best_score,

            "answer":
                best_answer,

            "results":
                best_results,

            "attempts":
                attempts,

            "memory_details":
                memory_details
        }