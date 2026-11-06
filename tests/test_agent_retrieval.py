import unittest
from unittest.mock import Mock

from src.agent.self_improving_agent import SelfImprovingRAGAgent


class TestAgentRetrieval(unittest.TestCase):

    def setUp(self):
        self.faiss_store = Mock()
        self.bm25_store = Mock()
        self.hybrid_store = Mock()
        self.reranker = Mock()
        self.embedding_model = Mock()

        self.agent = SelfImprovingRAGAgent(
            faiss_store=self.faiss_store,
            bm25_store=self.bm25_store,
            hybrid_store=self.hybrid_store,
            reranker=self.reranker,
            embedding_model=self.embedding_model,
            rewriter=None,
            generator=None,
            evaluator=None,
            memory=None
        )

        self.query = "original query"
        self.rewritten_query = "rewritten query"
        self.candidates = [{"text": "candidate"}]
        self.reranked = [{"text": "reranked result"}]

        self.hybrid_store.search.return_value = self.candidates
        self.faiss_store.search.return_value = self.candidates
        self.bm25_store.search.return_value = self.candidates
        self.reranker.rerank.return_value = self.reranked

    def test_hybrid_rewritten_uses_rewritten_query(self):
        result = self.agent.retrieve(
            self.query,
            self.rewritten_query,
            "hybrid_rewritten"
        )

        self.hybrid_store.search.assert_called_once_with(
            self.rewritten_query,
            top_k=10,
            candidate_k=15
        )
        self.reranker.rerank.assert_called_once_with(
            self.query,
            self.candidates,
            top_k=5
        )
        self.assertEqual(result, self.reranked)

    def test_hybrid_original_uses_original_query(self):
        self.agent.retrieve(
            self.query,
            self.rewritten_query,
            "hybrid_original"
        )

        self.hybrid_store.search.assert_called_once_with(
            self.query,
            top_k=10,
            candidate_k=15
        )

    def test_dense_rewritten_encodes_rewritten_query(self):
        self.embedding_model.encode.return_value = [[0.1, 0.2]]

        self.agent.retrieve(
            self.query,
            self.rewritten_query,
            "dense_rewritten"
        )

        self.embedding_model.encode.assert_called_once_with(
            [self.rewritten_query]
        )
        self.faiss_store.search.assert_called_once_with(
            [0.1, 0.2],
            top_k=10
        )

    def test_bm25_rewritten_uses_rewritten_query(self):
        self.agent.retrieve(
            self.query,
            self.rewritten_query,
            "bm25_rewritten"
        )

        self.bm25_store.search.assert_called_once_with(
            self.rewritten_query,
            top_k=10
        )

    def test_unknown_strategy_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.agent.retrieve(
                self.query,
                self.rewritten_query,
                "unknown_strategy"
            )

        self.reranker.rerank.assert_not_called()


if __name__ == "__main__":
    unittest.main()
