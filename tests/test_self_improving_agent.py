import unittest

from src.agent.self_improving_agent import SelfImprovingRAGAgent


class FakeRewriter:
    def rewrite(self, query):
        return f"rewritten: {query}"


class FakeMemory:
    def __init__(self, has_memory_match, initial_strategy):
        self.has_memory_match = has_memory_match
        self.initial_strategy = initial_strategy
        self.experiences = []

    def suggest_strategy(
        self,
        query,
        embedding_model,
        strategies,
        default_strategy
    ):
        return self.initial_strategy, {
            "has_memory_match": self.has_memory_match
        }

    def add(self, experience):
        self.experiences.append(experience)


class TestSelfImprovingRAGAgent(unittest.TestCase):

    def make_agent(
        self,
        has_memory_match=False,
        initial_strategy="hybrid_rewritten"
    ):
        memory = FakeMemory(
            has_memory_match,
            initial_strategy
        )

        agent = SelfImprovingRAGAgent(
            faiss_store=None,
            bm25_store=None,
            hybrid_store=None,
            reranker=None,
            embedding_model=None,
            rewriter=FakeRewriter(),
            generator=None,
            evaluator=None,
            memory=memory
        )

        return agent, memory

    def configure_attempts(self, agent, outcomes):
        calls = []

        def fake_run_attempt(query, rewritten_query, strategy):
            calls.append(strategy)

            score, success = outcomes[strategy]

            attempt = {
                "strategy": strategy,
                "overall_score": score,
                "success": success
            }

            answer = f"answer from {strategy}"
            results = [{"page": len(calls)}]

            return attempt, answer, results

        agent.run_attempt = fake_run_attempt

        return calls

    def test_no_memory_match_explores_all_strategies(self):
        agent, memory = self.make_agent(
            has_memory_match=False
        )

        expected = agent.strategies.copy()

        outcomes = {
            strategy: (0.7, False)
            for strategy in expected
        }

        calls = self.configure_attempts(
            agent,
            outcomes
        )

        result = agent.run("What is Python?")

        self.assertEqual(calls, expected)
        self.assertEqual(result["mode"], "explore")
        self.assertEqual(len(memory.experiences), 1)

    def test_exploitation_stops_after_preferred_strategy_succeeds(self):
        agent, memory = self.make_agent(
            has_memory_match=True,
            initial_strategy="hybrid_original"
        )

        outcomes = {
            "hybrid_original": (0.6, False),
            "hybrid_rewritten": (0.9, True),
            "dense_rewritten": (0.7, False),
            "bm25_rewritten": (0.7, False)
        }

        calls = self.configure_attempts(
            agent,
            outcomes
        )

        result = agent.run("What is Python?")

        self.assertEqual(
            calls,
            ["hybrid_original", "hybrid_rewritten"]
        )
        self.assertEqual(result["mode"], "exploit")
        self.assertEqual(result["best_strategy"], "hybrid_rewritten")

    def test_forced_exploration_runs_every_strategy(self):
        agent, memory = self.make_agent(
            has_memory_match=True,
            initial_strategy="bm25_rewritten"
        )

        expected = agent.strategies.copy()

        outcomes = {
            strategy: (0.9, True)
            for strategy in expected
        }

        calls = self.configure_attempts(
            agent,
            outcomes
        )

        result = agent.run(
            "What is Python?",
            force_explore=True
        )

        self.assertEqual(calls, expected)
        self.assertEqual(result["mode"], "explore_forced")

    def test_tied_strategies_are_preserved(self):
        agent, memory = self.make_agent(
            has_memory_match=False
        )

        outcomes = {
            "hybrid_rewritten": (0.9, True),
            "hybrid_original": (0.9, True),
            "dense_rewritten": (0.7, False),
            "bm25_rewritten": (0.6, False)
        }

        self.configure_attempts(
            agent,
            outcomes
        )

        result = agent.run("What is Python?")

        self.assertEqual(
            result["best_strategies"],
            ["hybrid_rewritten", "hybrid_original"]
        )
        self.assertTrue(result["best_strategy_tie"])
        self.assertEqual(
            result["best_strategy"],
            "hybrid_rewritten"
        )
        self.assertEqual(
            result["answer"],
            "answer from hybrid_rewritten"
        )

        stored = memory.experiences[0]

        self.assertEqual(
            stored["best_strategies"],
            ["hybrid_rewritten", "hybrid_original"]
        )

    def test_learn_false_does_not_store_experience(self):
        agent, memory = self.make_agent()

        outcomes = {
            strategy: (0.8, True)
            for strategy in agent.strategies
        }

        self.configure_attempts(
            agent,
            outcomes
        )

        agent.run(
            "What is Python?",
            learn=False
        )

        self.assertEqual(len(memory.experiences), 0)


if __name__ == "__main__":
    unittest.main()
