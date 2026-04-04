from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smart_ai_assistant.storage import FeedbackStore
from smart_ai_assistant.generation import _quality_score


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "feedback.db"
        self.store = FeedbackStore(self.database_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_ranking_prefers_best_votes_then_average_rating(self) -> None:
        responses = self.store.save_generated_responses(
            prompt="A robot helping humans in daily life",
            responses=[
                "Response one",
                "Response two",
                "Response three",
            ],
        )

        self.store.record_feedback(
            ratings_by_response_id={
                responses[0].id: 4,
                responses[1].id: 5,
                responses[2].id: 3,
            },
            best_response_id=responses[1].id,
        )
        self.store.record_feedback(
            ratings_by_response_id={
                responses[0].id: 5,
                responses[1].id: 4,
                responses[2].id: 2,
            },
            best_response_id=responses[0].id,
        )

        ranked = self.store.get_ranked_responses("A robot helping humans in daily life")

        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].id, responses[0].id)
        self.assertEqual(ranked[1].id, responses[1].id)
        self.assertEqual(ranked[2].id, responses[2].id)

    def test_invalid_rating_raises_value_error(self) -> None:
        responses = self.store.save_generated_responses(
            prompt="A cat wearing sunglasses on the beach",
            responses=["A bright beach scene"],
        )

        with self.assertRaises(ValueError):
            self.store.record_feedback(
                ratings_by_response_id={responses[0].id: 6},
                best_response_id=responses[0].id,
            )


class GenerationQualityTests(unittest.TestCase):
    def test_quality_score_prefers_relevant_scene_description(self) -> None:
        prompt = "A robot helping humans in daily life"
        strong = "A friendly robot carries groceries for an older man while guiding him across a bright city sidewalk. The scene feels practical, calm, and easy to imagine."
        weak = '"" Robot: An Indian girl making baby pets'

        self.assertGreater(_quality_score(prompt, strong), _quality_score(prompt, weak))


if __name__ == "__main__":
    unittest.main()
