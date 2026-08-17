from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from standards_rag.conversation_store import InMemoryConversationStore
from standards_rag.feedback import (
    MAX_COMMENT_CHARS,
    FeedbackError,
    FeedbackLog,
    InMemoryFeedbackStore,
)


def _log_with_a_saved_chat() -> tuple[FeedbackLog, str]:
    conversations = InMemoryConversationStore()
    record = conversations.create_conversation("user-1", title="New chat")
    conversations.append_turn(
        "user-1",
        record.conversation_id,
        question="Does my spec meet GM13?",
        answer="Your document allows 40 mil, GM13 does not. [1]",
        citations=[],
    )
    conversations.append_turn(
        "user-1",
        record.conversation_id,
        question="And the geotextile?",
        answer="D4595 governs wide-width tensile strength. [1]",
        citations=[],
    )
    log = FeedbackLog(InMemoryFeedbackStore(), conversations)
    return log, record.conversation_id


class RecordingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log, self.conversation_id = _log_with_a_saved_chat()

    def test_the_exchange_is_resolved_from_the_saved_chat_not_trusted_from_the_browser(self) -> None:
        record = self.log.record(
            user_id="user-1",
            user_email="Member@GSI.org",
            conversation_id=self.conversation_id,
            rating="down",
            answer="D4595 governs wide-width tensile strength. [1]",
        )

        self.assertEqual(record.question, "And the geotextile?")
        self.assertEqual(record.answer, "D4595 governs wide-width tensile strength. [1]")
        self.assertEqual(record.user_email, "member@gsi.org")

    def test_the_right_turn_is_found_when_a_chat_has_several(self) -> None:
        record = self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            answer="Your document allows 40 mil, GM13 does not. [1]",
        )
        self.assertEqual(record.question, "Does my spec meet GM13?")

    def test_an_answer_we_cannot_place_still_records_rather_than_being_lost(self) -> None:
        record = self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            answer="Some answer that was never saved.",
        )
        self.assertEqual(record.answer, "Some answer that was never saved.")
        self.assertEqual(record.question, "")

    def test_a_rating_we_do_not_recognise_is_refused(self) -> None:
        with self.assertRaises(FeedbackError):
            self.log.record(
                user_id="user-1",
                user_email="member@gsi.org",
                conversation_id=self.conversation_id,
                rating="sideways",
            )

    def test_a_rating_with_no_conversation_is_refused(self) -> None:
        with self.assertRaises(FeedbackError):
            self.log.record(
                user_id="user-1", user_email="member@gsi.org", conversation_id="", rating="down"
            )

    def test_a_long_comment_is_clipped(self) -> None:
        record = self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            comment="x" * (MAX_COMMENT_CHARS + 500),
        )
        self.assertLessEqual(len(record.comment), MAX_COMMENT_CHARS + 1)


class OneOpinionOneRowTests(unittest.TestCase):
    """A thumbs-down and the comment that follows it are one opinion."""

    def setUp(self) -> None:
        self.log, self.conversation_id = _log_with_a_saved_chat()

    def test_a_comment_lands_on_the_rating_it_belongs_to(self) -> None:
        first = self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            answer="And the geotextile?",
        )
        self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            comment="It missed clause 2.4.",
            feedback_id=first.feedback_id,
            created_at=first.created_at,
        )

        rows = self.log.conversations()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 1)
        self.assertEqual(rows[0]["items"][0]["comment"], "It missed clause 2.4.")

    def test_two_separate_ratings_stay_two_rows(self) -> None:
        for _ in range(2):
            self.log.record(
                user_id="user-1",
                user_email="member@gsi.org",
                conversation_id=self.conversation_id,
                rating="down",
            )
        self.assertEqual(self.log.summary()["total"], 2)


class ReadingBackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.log, self.conversation_id = _log_with_a_saved_chat()

    def test_feedback_is_grouped_under_the_chat_it_came_from(self) -> None:
        self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=self.conversation_id,
            rating="down",
            comment="Wrong.",
        )
        self.log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id="other-chat",
            rating="up",
        )

        rows = self.log.conversations()
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["conversation_id"] for row in rows}, {self.conversation_id, "other-chat"})

    def test_the_group_is_titled_with_the_chat_s_own_name(self) -> None:
        # The admin's tab reads "Conversation about <title>", so the title has
        # to be the one the chat already generated for itself.
        conversations = InMemoryConversationStore()
        record = conversations.create_conversation("user-1", title="GM13 Thickness")
        conversations.append_turn("user-1", record.conversation_id, question="q", answer="a", citations=[])
        record.title = "GM13 Thickness"
        log = FeedbackLog(InMemoryFeedbackStore(), conversations)
        log.record(
            user_id="user-1",
            user_email="member@gsi.org",
            conversation_id=record.conversation_id,
            rating="down",
            answer="a",
        )
        self.assertEqual(log.conversations()[0]["title"], "GM13 Thickness")

    def test_dislikes_are_counted_separately_from_likes(self) -> None:
        self.log.record(
            user_id="user-1", user_email="m@g.org", conversation_id=self.conversation_id, rating="down"
        )
        self.log.record(
            user_id="user-1", user_email="m@g.org", conversation_id=self.conversation_id, rating="up"
        )

        summary = self.log.summary()
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["down_count"], 1)
        self.assertEqual(summary["up_count"], 1)
        self.assertEqual(self.log.conversations()[0]["down_count"], 1)

    def test_a_chat_we_cannot_read_still_records_the_rating(self) -> None:
        class Broken:
            def get_conversation(self, *_args, **_kwargs):
                raise RuntimeError("DynamoDB is having a moment")

        log = FeedbackLog(InMemoryFeedbackStore(), Broken())
        record = log.record(
            user_id="user-1",
            user_email="m@g.org",
            conversation_id="c1",
            rating="down",
            comment="Bad answer.",
            answer="the answer",
        )
        self.assertEqual(record.comment, "Bad answer.")
        self.assertEqual(record.conversation_title, "Untitled chat")

    def test_nothing_rated_reads_as_nothing(self) -> None:
        self.assertEqual(self.log.conversations(), [])
        self.assertEqual(self.log.summary()["total"], 0)


if __name__ == "__main__":
    unittest.main()
