"""Live end-to-end tests against chat.deepseek.com.

Skipped unless DS_LIVE=1 and real credentials are available. Each run consumes
a couple of chat turns on the account behind the `tokens` file, so this is opt-in:

    DS_LIVE=1 python3 -m unittest discover -s tests -v

Credentials come from `tokens` (or DS_SESSION_ID / AUTHORIZATION_TOKEN / DS_COOKIES),
in the format described in the README.
"""

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LIVE = os.environ.get("DS_LIVE") == "1"
TOKENS = REPO_ROOT / "tokens"
HAVE_CREDS = LIVE and (
    (TOKENS.exists() or os.environ.get("DS_SESSION_ID"))
    and (os.environ.get("AUTHORIZATION_TOKEN") or TOKENS.exists())
)


@unittest.skipUnless(HAVE_CREDS, "set DS_LIVE=1 and provide credentials to run")
class TestLive(unittest.TestCase):
    def _chat(self):
        from DeepSeekAPI.DeepSeekChat.main import DeepSeekChat, load_tokens
        session_id, token, cookies = load_tokens(str(TOKENS))
        return DeepSeekChat(session_id, token, cookies=cookies)

    def test_plain_turn(self):
        chat = self._chat()
        result = chat.send_message("hi", printing=False, thinking_enabled=False,
                                   search_enabled=False, model_type="default")
        self.assertTrue(result["ok"], result.get("content"))
        content = result["content"]
        self.assertTrue(content["response"].strip())
        self.assertEqual(content["model_type"], "default")

    def test_reasoning_turn_uses_null_model_type(self):
        chat = self._chat()
        result = chat.send_message("say OK and nothing else", printing=False,
                                   thinking_enabled=True, search_enabled=False,
                                   model_type=None)
        self.assertTrue(result["ok"], result.get("content"))
        content = result["content"]
        self.assertIsNone(content["model_type"])
        self.assertTrue(content["response"].strip())
        self.assertTrue(content["thought"].strip())
        self.assertGreater(content["thinktime"], 0)

    def test_parent_message_id_threads(self):
        chat = self._chat()
        first = chat.send_message("hi", printing=False, model_type="default")
        self.assertTrue(first["ok"])
        self.assertIsNotNone(chat.parent_message_id)
        second = chat.send_message("say OK and nothing else", printing=False,
                                   model_type=None, thinking_enabled=True)
        self.assertTrue(second["ok"], second.get("content"))


if __name__ == "__main__":
    unittest.main()
