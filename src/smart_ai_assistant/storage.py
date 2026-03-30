from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GeneratedResponse:
    id: int
    prompt: str
    response_text: str


@dataclass(frozen=True)
class RankedResponse:
    id: int
    prompt: str
    response_text: str
    average_rating: float
    rating_count: int
    best_vote_count: int

    @property
    def reward_score(self) -> float:
        return round((self.best_vote_count * 2.0) + self.average_rating, 2)

    def as_table_row(self) -> dict[str, object]:
        return {
            "Response ID": self.id,
            "Average Rating": round(self.average_rating, 2),
            "Ratings": self.rating_count,
            "Best Votes": self.best_vote_count,
            "Reward Proxy": self.reward_score,
            "Response Preview": self.response_text[:100] + ("..." if len(self.response_text) > 100 else ""),
        }


class FeedbackStore:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id INTEGER NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    selected_best INTEGER NOT NULL DEFAULT 0 CHECK (selected_best IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (response_id) REFERENCES responses(id) ON DELETE CASCADE
                );
                """
            )

    def save_generated_responses(self, prompt: str, responses: Iterable[str]) -> list[GeneratedResponse]:
        cleaned_responses = [response.strip() for response in responses if response.strip()]
        if not cleaned_responses:
            raise ValueError("At least one non-empty response is required.")

        created: list[GeneratedResponse] = []
        with self._connect() as connection:
            cursor = connection.cursor()
            for response_text in cleaned_responses:
                cursor.execute(
                    "INSERT INTO responses (prompt, response_text) VALUES (?, ?)",
                    (prompt, response_text),
                )
                created.append(
                    GeneratedResponse(
                        id=int(cursor.lastrowid),
                        prompt=prompt,
                        response_text=response_text,
                    )
                )
        return created

    def record_feedback(self, ratings_by_response_id: dict[int, int], best_response_id: int | None) -> None:
        if not ratings_by_response_id:
            raise ValueError("Feedback requires at least one rating.")

        for response_id, rating in ratings_by_response_id.items():
            if rating < 1 or rating > 5:
                raise ValueError(f"Rating for response {response_id} must be between 1 and 5.")

        with self._connect() as connection:
            cursor = connection.cursor()
            for response_id, rating in ratings_by_response_id.items():
                cursor.execute(
                    """
                    INSERT INTO feedback (response_id, rating, selected_best)
                    VALUES (?, ?, ?)
                    """,
                    (response_id, rating, 1 if best_response_id == response_id else 0),
                )

    def get_ranked_responses(self, prompt: str) -> list[RankedResponse]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    responses.id,
                    responses.prompt,
                    responses.response_text,
                    COALESCE(AVG(feedback.rating), 0.0) AS average_rating,
                    COUNT(feedback.rating) AS rating_count,
                    COALESCE(SUM(feedback.selected_best), 0) AS best_vote_count
                FROM responses
                LEFT JOIN feedback ON feedback.response_id = responses.id
                WHERE responses.prompt = ?
                GROUP BY responses.id
                ORDER BY best_vote_count DESC, average_rating DESC, rating_count DESC, responses.id ASC
                """,
                (prompt,),
            ).fetchall()

        return [
            RankedResponse(
                id=int(row["id"]),
                prompt=str(row["prompt"]),
                response_text=str(row["response_text"]),
                average_rating=float(row["average_rating"]),
                rating_count=int(row["rating_count"]),
                best_vote_count=int(row["best_vote_count"]),
            )
            for row in rows
        ]

    def get_dashboard_metrics(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM responses) AS response_count,
                    (SELECT COUNT(*) FROM feedback) AS feedback_count,
                    (SELECT COUNT(DISTINCT prompt) FROM responses) AS prompt_count
                """
            ).fetchone()

        return {
            "response_count": int(row["response_count"]),
            "feedback_count": int(row["feedback_count"]),
            "prompt_count": int(row["prompt_count"]),
        }
