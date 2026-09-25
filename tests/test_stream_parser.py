"""SSE stream parser tests.

The fixtures under tests/fixtures/ are real captured streams from the web
client (x-client-version 2.5.0), so the assertions below pin the exact text
that must be assembled for both plain and reasoning turns.
"""

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from DeepSeekAPI.DeepSeekChat.main import (  # noqa: E402
    CLIENT_VERSION,
    VALID_MODEL_TYPES,
    DeepSeekChat,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def parse(lines):
    """Run the parser without touching the network."""
    chat = DeepSeekChat.__new__(DeepSeekChat)  # _parse_stream does not use self
    captured = []
    state = chat._parse_stream(
        iter(lines),
        emit=lambda kind, text: captured.append((kind, text)),
        display=lambda text: None,
        show_markers=False,
    )
    return state, captured


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8").split("\n")


class TestRealCapture(unittest.TestCase):
    def test_no_thinking(self):
        state, captured = parse(fixture("stream_nothink.txt"))
        self.assertEqual(state["respond"], "Hello! How can I help you today?")
        self.assertEqual(state["think"], "")
        self.assertEqual(state["title"], "Greeting")
        self.assertEqual(state["msgid"], 2)
        self.assertEqual(state["tokencount"], 42)
        self.assertEqual({kind for kind, _ in captured}, {"response", "status"})

    def test_with_thinking(self):
        state, captured = parse(fixture("stream_think.txt"))
        self.assertEqual(
            state["think"],
            'We need answer greeting. User says "hi". Need respond friendly. '
            'Since no task. Just greet. Maybe ask how can help. '
            'Need in English. concise.',
        )
        self.assertEqual(state["respond"], "Hi! How can I help you today?")
        self.assertNotIn("concise.", state["respond"])
        self.assertEqual(state["thinktime"], 0.827795957)
        self.assertEqual(state["tokencount"], 67)
        self.assertEqual({kind for kind, _ in captured}, {"think", "response", "status"})

    def test_first_fragment_is_not_dropped(self):
        """Regression: the initial fragment's text arrives in the first object."""
        state, _ = parse(fixture("stream_think.txt"))
        self.assertTrue(state["respond"].startswith("Hi"))
        self.assertTrue(state["think"].startswith("We"))


class TestLegacyProtocol(unittest.TestCase):
    """Older streams switch modes with {"type": ...} dicts."""

    LEGACY = """event: ready
data: {"request_message_id": 1, "response_message_id": 2}

event: update_session
data: {"type": "THINK", "content": "thinking about it"}

event: message
data: {"type": "RESPONSE"}

event: message
data: {"p": "/message/content", "v": "Hel"}

event: message
data: {"p": "/message/response", "v": "lo"}

event: update_session
data: {"message_id": 2, "parent_id": 1}

event: title
data: {"content": "A title"}

event: close
data: {}

event: message
data: {"p": "/message/content", "v": "AFTER-CLOSE"}

: heartbeat
data: [DONE]
"""

    def setUp(self):
        self.state, self.captured = parse(self.LEGACY.split("\n"))

    def test_mode_switch_still_works(self):
        self.assertEqual(self.state["think"], "thinking about it")
        self.assertEqual(self.state["respond"], "Hello")

    def test_stream_stops_at_close(self):
        self.assertNotIn("AFTER-CLOSE", self.state["respond"])

    def test_metadata(self):
        self.assertEqual(self.state["title"], "A title")
        self.assertEqual(self.state["msgid"], 2)


class TestTolerance(unittest.TestCase):
    """Unknown events and fields must be skipped, never fatal."""

    NOISY = """event: ready
data: {"request_message_id": 1, "response_message_id": 2}

event: something_unheard_of
data: {"x": 1}

event: update_session
data: {"updated_at": 1790331638.8}

event: message
data: {"type": "RESPONSE", "content": "ok"}

event: message
data: {"p": "response/brand_new_field", "v": "?"}

event: message
data: {"p": "response/fragments/-1/content", "o": "APPEND", "v": "!"}

event: message
data: not json at all

event: close
data: {}
"""

    def test_unknown_shapes_are_skipped(self):
        state, _ = parse(self.NOISY.split("\n"))
        self.assertEqual(state["respond"], "ok!")

    def test_markers_never_leak_into_on_delta(self):
        _, captured = parse(self.NOISY.split("\n"))
        self.assertTrue(all(not text.startswith("\n\n-----") for _, text in captured))


class TestHeaderProfile(unittest.TestCase):
    def test_matches_observed_client(self):
        chat = DeepSeekChat("sid", "Bearer tok")
        self.assertEqual(chat.headers["x-client-version"], CLIENT_VERSION)
        self.assertNotIn("x-app-version", chat.headers)
        self.assertFalse(any(k.startswith("x-debug") for k in chat.headers))
        self.assertEqual(chat.headers["origin"], "https://chat.deepseek.com")
        self.assertTrue(chat.headers["x-device-id"])
        self.assertEqual(chat.headers["sec-fetch-site"], "same-origin")

    def test_referer_spelling(self):
        chat = DeepSeekChat("sid", "Bearer tok")
        self.assertEqual(chat._referer(), "https://chat.deepseek.com/")
        chat.chat_session_id = "abc"
        self.assertEqual(chat._referer(), "https://chat.deepseek.com/a/chat/s/abc")

    def test_cookie_jar(self):
        chat = DeepSeekChat("sid", "Bearer tok", cookies="a=1; ds_session_id=xyz")
        self.assertEqual(chat.session.cookies.get("a"), "1")
        self.assertEqual(chat.session.cookies.get("ds_session_id"), "xyz")

    def test_model_type_values(self):
        self.assertIn(None, VALID_MODEL_TYPES)
        self.assertEqual(VALID_MODEL_TYPES, (None, "default", "expert"))


if __name__ == "__main__":
    unittest.main()
