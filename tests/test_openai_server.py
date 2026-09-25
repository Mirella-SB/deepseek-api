"""OpenAI-compatible HTTP layer tests.

The upstream client is stubbed out, so these run offline and assert request
mapping, response shapes, streaming framing and error handling.
"""

import json
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# server.py resolves credentials at import time; give it dummies.
os.environ.setdefault("DS_SESSION_ID", "test-session")
os.environ.setdefault("AUTHORIZATION_TOKEN", "Bearer test-token")

import server  # noqa: E402
from DeepSeekAPI.DeepSeekChat.main import DeepSeekChat  # noqa: E402


def stub_send_message(self, message, printing=None, thinking_enabled=False,
                      search_enabled=False, model_type="default", on_delta=None):
    stub_send_message.calls.append({
        "message": message, "thinking": thinking_enabled,
        "search": search_enabled, "model_type": model_type,
    })
    for kind, text in (("think", "hmm "), ("think", "thinking"),
                       ("response", "Hi"), ("response", " there")):
        if on_delta:
            on_delta(kind, text)
    return {"ok": True, "content": {
        "thinking_enabled": thinking_enabled, "search_enabled": search_enabled,
        "model_type": model_type, "response": "Hi there",
        "thought": "hmm thinking" if thinking_enabled else "",
        "thinktime": 0.5 if thinking_enabled else 0,
        "title": "t",
    }}


stub_send_message.calls = []


class ServerTestCase(unittest.TestCase):
    def setUp(self):
        stub_send_message.calls = []
        self._original = DeepSeekChat.send_message
        DeepSeekChat.send_message = stub_send_message
        self.client = server.app.test_client()

    def tearDown(self):
        DeepSeekChat.send_message = self._original

    def post(self, **body):
        return self.client.post("/v1/chat/completions", json=body)


class TestModels(ServerTestCase):
    def test_lists_models(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [model["id"] for model in response.get_json()["data"]],
            ["deepseek-v3", "deepseek-r1", "deepseek-v4", "deepseek-r4"],
        )

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})


class TestModelValidation(ServerTestCase):
    def test_unknown_model_is_rejected(self):
        response = self.post(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"]["code"], "model_not_found")
        self.assertEqual(stub_send_message.calls, [])

    def test_known_aliases(self):
        for model in ("deepseek-v3", "deepseek-chat", "deepseek-reasoner"):
            self.post(model=model, messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(stub_send_message.calls), 3)


class TestMapping(ServerTestCase):
    def test_v3_maps_to_default_without_thinking(self):
        self.post(model="deepseek-v3", messages=[{"role": "user", "content": "hi"}])
        call = stub_send_message.calls[-1]
        self.assertEqual(call["model_type"], "default")
        self.assertFalse(call["thinking"])

    def test_r1_maps_to_null_model_type(self):
        """The web client sends model_type null on reasoning turns."""
        self.post(model="deepseek-r1", messages=[{"role": "user", "content": "hi"}])
        call = stub_send_message.calls[-1]
        self.assertIsNone(call["model_type"])
        self.assertTrue(call["thinking"])

    def test_prompt_comes_from_last_user_message(self):
        self.post(model="deepseek-v3", messages=[
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ])
        self.assertEqual(stub_send_message.calls[-1]["message"], "second")

    def test_search_enabled_passes_through(self):
        self.post(model="deepseek-v3", search_enabled=True,
                  messages=[{"role": "user", "content": "hi"}])
        self.assertTrue(stub_send_message.calls[-1]["search"])


class TestNonStreaming(ServerTestCase):
    def test_response_shape(self):
        response = self.post(model="deepseek-v3",
                             messages=[{"role": "user", "content": "hi"}])
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hi there")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("reasoning_content", body["choices"][0]["message"])

    def test_reasoning_content_for_r1(self):
        response = self.post(model="deepseek-r1",
                             messages=[{"role": "user", "content": "hi"}])
        message = response.get_json()["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "hmm thinking")
        self.assertEqual(message["content"], "Hi there")


class TestStreaming(ServerTestCase):
    def setUp(self):
        super().setUp()
        response = self.post(model="deepseek-v3", stream=True,
                             messages=[{"role": "user", "content": "hi"}])
        self.raw = response.get_data(as_text=True)
        self.content_type = response.headers["Content-Type"]

    def test_event_stream(self):
        self.assertTrue(self.content_type.startswith("text/event-stream"))
        self.assertTrue(self.raw.rstrip().endswith("data: [DONE]"))

    def test_chunks(self):
        chunks = [json.loads(line[6:]) for line in self.raw.split("\n")
                  if line.startswith("data: ") and line.strip() != "data: [DONE]"]
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"].get("role"), "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in chunks)
        self.assertEqual(text, "Hi there")
        self.assertEqual(reasoning, "hmm thinking")


class TestErrors(ServerTestCase):
    def test_upstream_error_becomes_502(self):
        def boom(*args, **kwargs):
            return {"ok": False, "content": "DeepSeek error 40300: MISSING_HEADER"}
        DeepSeekChat.send_message = boom
        response = self.post(model="deepseek-v3",
                             messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.get_json()["error"]["type"], "upstream_error")


if __name__ == "__main__":
    unittest.main()
