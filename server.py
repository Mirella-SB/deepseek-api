#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import json
import os
import queue
import threading
import time
import uuid

from flask import Flask, Response, jsonify, request, stream_with_context

from DeepSeekAPI import DeepSeekChat
from DeepSeekAPI.DeepSeekChat.main import load_tokens

app = Flask(__name__)

DS_SESSION_ID, AUTHORIZATION_TOKEN, DS_COOKIES = load_tokens()
DEVICE_ID = os.environ.get("DS_DEVICE_ID")


def _chat():
    return DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN,
                        cookies=DS_COOKIES, device_id=DEVICE_ID)


# OpenAI-style model ids -> DeepSeek web flags.
#
# Observed on the wire: non-reasoning turns send model_type "default", reasoning
# turns send model_type null. "expert" is preserved for compatibility but is not
# backed by captured traffic.
KNOWN_MODELS = {
    "deepseek-v3": {"model_type": "default", "thinking_enabled": False},
    "deepseek-chat": {"model_type": "default", "thinking_enabled": False},
    "deepseek-r1": {"model_type": None, "thinking_enabled": True},
    "deepseek-reasoner": {"model_type": None, "thinking_enabled": True},
    "deepseek-v4": {"model_type": "expert", "thinking_enabled": False},
    "deepseek-r4": {"model_type": "expert", "thinking_enabled": True},
}


def get_model_config(model: str):
    """Map an OpenAI-style model name to (model_type, thinking_enabled)."""
    key = (model or "").strip().lower()
    if key in KNOWN_MODELS:
        cfg = KNOWN_MODELS[key]
        return cfg["model_type"], cfg["thinking_enabled"]
    thinking_enabled = any(tag in key for tag in ("r1", "r4", "reasoning", "reasoner"))
    expert = any(tag in key for tag in ("v4", "r4", "expert"))
    if expert:
        model_type = "expert"
    elif thinking_enabled:
        model_type = None
    else:
        model_type = "default"
    return model_type, thinking_enabled


def model_error(model):
    return jsonify({
        "error": {
            "message": f"The model '{model}' does not exist or you do not have access to it.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }), 404


def _user_prompt(messages):
    """The web API takes a single prompt; use the last user message."""
    for message in reversed(messages or []):
        if message.get("role") == "user" and message.get("content"):
            return message["content"]
    return (messages[-1]["content"] if messages else "")


def _chat_args(data):
    model = data.get("model", "deepseek-v3")
    if model not in KNOWN_MODELS:
        return None, model_error(model)
    model_type, thinking_enabled = get_model_config(model)
    return {
        "model": model,
        "model_type": model_type,
        "thinking_enabled": thinking_enabled,
        "search_enabled": bool(data.get("search_enabled", False)),
        "prompt": _user_prompt(data.get("messages", [])),
    }, None


def chat_non_streaming(args):
    chat = _chat()
    result = chat.send_message(
        args["prompt"],
        printing=False,
        thinking_enabled=args["thinking_enabled"],
        search_enabled=args["search_enabled"],
        model_type=args["model_type"],
    )
    if not result or not result.get("ok"):
        return jsonify({
            "error": {
                "message": str(result.get("content") if result else "no response"),
                "type": "upstream_error",
                "code": "deepseek_error",
            }
        }), 502

    content = result["content"]
    message = {"role": "assistant", "content": content.get("response", "")}
    if args["thinking_enabled"]:
        message["reasoning_content"] = content.get("thought", "")

    return jsonify({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": args["model"],
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": content.get("tokencount") or 0,
            "total_tokens": content.get("tokencount") or 0,
        },
    })


def chat_streaming(args):
    """Stream SSE deltas as they are produced (real streaming, not a replay)."""
    chunks = queue.Queue()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def on_delta(kind, text):
        if kind in ("response", "think", "status", "search"):
            chunks.put((kind, text))

    def run():
        try:
            chat = _chat()
            result = chat.send_message(
                args["prompt"],
                printing=False,
                thinking_enabled=args["thinking_enabled"],
                search_enabled=args["search_enabled"],
                model_type=args["model_type"],
                on_delta=on_delta,
            )
            chunks.put(("__done__", result))
        except Exception as error:  # noqa: BLE001 - surfaced to the client
            chunks.put(("__error__", str(error)))

    threading.Thread(target=run, daemon=True).start()

    def envelope(delta, finish_reason=None):
        return "data: " + json.dumps({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": args["model"],
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }, ensure_ascii=False) + "\n\n"

    def generate():
        yield envelope({"role": "assistant", "content": ""})
        while True:
            kind, payload = chunks.get()
            if kind == "__done__":
                ok = isinstance(payload, dict) and payload.get("ok")
                if not ok:
                    yield envelope({"content": f"\n[error] {payload}"})
                yield envelope({}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return
            if kind == "__error__":
                yield envelope({"content": f"\n[error] {payload}"})
                yield envelope({}, finish_reason="stop")
                yield "data: [DONE]\n\n"
                return
            if kind == "think":
                yield envelope({"reasoning_content": payload})
            elif kind == "response":
                yield envelope({"content": payload})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive'},
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json or {}
    args, error = _chat_args(data)
    if error:
        return error
    if data.get("stream"):
        return chat_streaming(args)
    return chat_non_streaming(args)


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": description,
            }
            for model_id, description in (
                ("deepseek-v3", "DeepSeek V3 - fast responses without extended thinking"),
                ("deepseek-r1", "DeepSeek R1 - reasoning model with extended thinking"),
                ("deepseek-v4", "DeepSeek V4 - expert model without extended thinking"),
                ("deepseek-r4", "DeepSeek R4 - expert reasoning model with extended thinking"),
            )
        ],
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
