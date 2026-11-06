import unittest

from src.agent.self_improving_agent import SelfImprovingRAGAgent


class FakeGenerator:
    def __init__(self):
        self.regeneration_calls = 0

    def generate(self, query, results):
        return "initial answer"

    def regenerate(self, query, results, evaluation):
        self.regeneration_calls += 1
        return "regenerated answer"


class FakeEvaluator:
    def __init__(self, retrieval_eval, answer_evals):
        self.retrieval_eval = retrieval_eval
        self.answer_evals = answer_evals

    def evaluate_retrieval(self, query, results):
        return self.retrieval_eval

    def evaluate_answer(self, query, results, answer):
        return self.answer_evals.pop(0)


class TestAgentAttempt(unittest.TestCase):

    def make_agent(self, retrieval_eval, answer_evals):
        agent = SelfImprovingRAGAgent(
            faiss_store=None,
            bm25_store=None,
            hybrid_store=None,
            reranker=None,
            embedding_model=None,
            rewriter=None,
            generator=FakeGenerator(),
            evaluator=FakeEvaluator(
                retrieval_eval,
                answer_evals
            ),
            memory=None
        )

        results = [
            {"page": 12},
            {"page": 14}
        ]

        agent.retrieve = lambda query, rewritten_query, strategy: results

        return agent

    def retrieval_evaluation(
        self,
        retrieval_relevance=0.8,
        relevant_fraction=0.8
    ):
        return {
            "retrieval_relevance": retrieval_relevance,
            "relevant_fraction": relevant_fraction,
            "direct_fraction": 0.6,
            "chunks": []
        }

    def answer_evaluation(self, relevance, faithfulness):
        return {
            "answer_relevance": relevance,
            "faithfulness": faithfulness
        }

    def test_threshold_boundaries_are_successful_without_regeneration(self):
        agent = self.make_agent(
            self.retrieval_evaluation(),
            [self.answer_evaluation(0.75, 0.8)]
        )

        attempt, answer, results = agent.run_attempt(
            "What is Python?",
            "Python explanation",
            "hybrid_rewritten"
        )

        self.assertTrue(attempt["success"])
        self.assertFalse(attempt["regeneration_attempted"])
        self.assertFalse(attempt["regenerated"])
        self.assertEqual(agent.generator.regeneration_calls, 0)
        self.assertEqual(answer, "initial answer")
        self.assertEqual(attempt["diagnosis"], "success")
        self.assertEqual(attempt["retrieved_pages"], [12, 14])
        self.assertAlmostEqual(attempt["overall_score"], 0.785)

    def test_better_regenerated_answer_is_accepted(self):
        agent = self.make_agent(
            self.retrieval_evaluation(0.9),
            [
                self.answer_evaluation(0.6, 0.7),
                self.answer_evaluation(0.8, 0.85)
            ]
        )

        attempt, answer, results = agent.run_attempt(
            "What is Python?",
            "Python explanation",
            "hybrid_rewritten"
        )

        self.assertTrue(attempt["regeneration_attempted"])
        self.assertTrue(attempt["regenerated"])
        self.assertTrue(attempt["success"])
        self.assertEqual(answer, "regenerated answer")
        self.assertEqual(agent.generator.regeneration_calls, 1)
        self.assertAlmostEqual(attempt["overall_score"], 0.855)

    def test_worse_regenerated_answer_is_rejected(self):
        agent = self.make_agent(
            self.retrieval_evaluation(),
            [
                self.answer_evaluation(0.7, 0.75),
                self.answer_evaluation(0.72, 0.7)
            ]
        )

        attempt, answer, results = agent.run_attempt(
            "What is Python?",
            "Python explanation",
            "hybrid_rewritten"
        )

        self.assertTrue(attempt["regeneration_attempted"])
        self.assertFalse(attempt["regenerated"])
        self.assertFalse(attempt["success"])
        self.assertEqual(answer, "initial answer")
        self.assertEqual(attempt["answer_relevance"], 0.7)
        self.assertEqual(attempt["faithfulness"], 0.75)
        self.assertEqual(attempt["diagnosis"], "grounding_failure")

    def test_retrieval_failure_diagnosis(self):
        agent = self.make_agent(
            self.retrieval_evaluation(
                retrieval_relevance=0.4,
                relevant_fraction=0.5
            ),
            [
                self.answer_evaluation(0.7, 0.9),
                self.answer_evaluation(0.7, 0.9)
            ]
        )

        attempt, answer, results = agent.run_attempt(
            "What is Python?",
            "Python explanation",
            "bm25_rewritten"
        )

        self.assertEqual(
            attempt["diagnosis"],
            "retrieval_failure"
        )
        self.assertFalse(attempt["success"])


if __name__ == "__main__":
    unittest.main()
