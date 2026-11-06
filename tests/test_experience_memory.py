import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.memory.experience_memory import ExperienceMemory


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.tile(
            np.array([[1.0, 0.0]]),
            (len(texts), 1)
        )


def make_experience(query):
    return {
        "query": query,
        "best_strategy": "hybrid_rewritten",
        "attempts": [
            {
                "strategy": "hybrid_rewritten",
                "overall_score": 0.9,
                "success": True
            }
        ]
    }


class ExperienceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.path = Path(self.temp_dir.name) / "experiences.json"
        self.memory = ExperienceMemory(path=self.path)

        self.strategies = [
            "hybrid_rewritten",
            "dense_rewritten"
        ]

    def test_empty_memory_returns_default_strategy(self):
        model = FakeEmbeddingModel()

        strategy, details = self.memory.suggest_strategy(
            "new query",
            model,
            self.strategies,
            "hybrid_rewritten"
        )

        self.assertEqual(strategy, "hybrid_rewritten")
        self.assertFalse(details["has_memory_match"])
        self.assertEqual(model.calls, [])

    def test_historical_embeddings_are_cached_and_extended(self):
        self.memory.experiences = [
            make_experience("question one"),
            make_experience("question two")
        ]

        model = FakeEmbeddingModel()

        self.memory.suggest_strategy(
            "first query", model, self.strategies, "hybrid_rewritten"
        )

        self.memory.suggest_strategy(
            "second query", model, self.strategies, "hybrid_rewritten"
        )

        self.memory.add(
            make_experience("question three")
        )

        self.memory.suggest_strategy(
            "third query", model, self.strategies, "hybrid_rewritten"
        )

        self.assertEqual(
            model.calls,
            [
                ["question one", "question two"],
                ["first query"],
                ["second query"],
                ["question three"],
                ["third query"]
            ]
        )

    def test_cache_rebuilds_when_embedding_model_changes(self):
        self.memory.experiences = [
            make_experience("question one")
        ]

        first_model = FakeEmbeddingModel()
        second_model = FakeEmbeddingModel()

        self.memory.suggest_strategy(
            "first query", first_model,
            self.strategies, "hybrid_rewritten"
        )

        self.memory.suggest_strategy(
            "second query", second_model,
            self.strategies, "hybrid_rewritten"
        )

        self.assertEqual(len(first_model.calls), 2)
        self.assertEqual(
            second_model.calls,
            [
                ["question one"],
                ["second query"]
            ]
        )

    def test_cache_rebuilds_when_historical_query_changes(self):
        self.memory.experiences = [
            make_experience("original question")
        ]

        model = FakeEmbeddingModel()

        self.memory.suggest_strategy(
            "first query", model, self.strategies, "hybrid_rewritten"
        )

        self.memory.experiences[0]["query"] = "updated question"

        self.memory.suggest_strategy(
            "second query", model, self.strategies, "hybrid_rewritten"
        )

        self.assertEqual(
            model.calls,
            [
                ["original question"],
                ["first query"],
                ["updated question"],
                ["second query"]
            ]
        )

    def test_experiences_persist_to_json(self):
        experience = make_experience("saved question")
        self.memory.add(experience)

        reloaded_memory = ExperienceMemory(path=self.path)

        self.assertEqual(
            reloaded_memory.experiences,
            [experience]
        )


if __name__ == "__main__":
    unittest.main()
