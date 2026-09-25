#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API - Provides an unofficial API for DeepSeek by reverse-engineering its web interface.
Copyright (C) 2025 smkttl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from traceback import format_exc as backtrace

import requests

# --- Client profile, matched against real traffic (chat.deepseek.com web 2.5.0) ---
CLIENT_VERSION = "2.5.0"
CLIENT_LOCALE = "en_US"
CLIENT_BUNDLE_ID = "com.deepseek.chat"
USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Mobile Safari/537.36"
)
SEC_CH_UA = '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"'

# Observed values for `model_type` are "default" and null (null is what the web
# client sends on reasoning turns). "expert" is accepted for forward-compat.
VALID_MODEL_TYPES = (None, "default", "expert")

POW_TARGET_PATH = "/api/v0/chat/completion"

# Events that carry incremental stream payloads.
PAYLOAD_EVENTS = ("update_session", "message")
# Events that terminate the stream.
CLOSE_EVENTS = ("close", "finish")

# Fragment types -> accumulator keys. THINK/RESPONSE are the ones observed.
FRAGMENT_SINKS = {
    "THINK": "think",
    "RESPONSE": "response",
    "SEARCH": "search",
    "TIP": "tip",
}

VERBOSE = os.environ.get("DS_VERBOSE") == "1"


def _warn(message):
    if VERBOSE:
        print(f"[DeepSeekChat] {message}", file=sys.stderr, flush=True)


def _tz_offset_seconds():
    """Seconds east of UTC. The web client sends 28800 for UTC+8."""
    offset = datetime.now().astimezone().utcoffset() or timedelta(0)
    return int(offset.total_seconds())


def _as_number(value):
    """Stream fields like elapsed_secs arrive as strings; make them numeric."""
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _first_present(obj, *paths):
    """Return the first non-None value found at any of the dotted paths."""
    for path in paths:
        node = obj
        for key in path.split('.'):
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            return node
    return None


def load_tokens(path="tokens"):
    """Read (ds_session_id, authorization_token, cookie_string).

    The `tokens` file is up to three lines:
        <ds_session_id>
        <authorization_token>
        <optional raw cookie string, e.g. "a=b; c=d">

    Environment variables DS_SESSION_ID / AUTHORIZATION_TOKEN / DS_COOKIES
    take precedence and are used when the file is absent.
    """
    ds_session_id = os.environ.get("DS_SESSION_ID")
    authorization_token = os.environ.get("AUTHORIZATION_TOKEN")
    cookies = os.environ.get("DS_COOKIES")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.read().split("\n")]
        ds_session_id = ds_session_id or (lines[0] if lines else "")
        authorization_token = authorization_token or (lines[1] if len(lines) > 1 else "")
        cookies = cookies or (lines[2] if len(lines) > 2 and lines[2] else None)
    if not ds_session_id or not authorization_token:
        raise ValueError(
            "Tokens not found. Set DS_SESSION_ID and AUTHORIZATION_TOKEN, "
            "or create a 'tokens' file (see load_tokens())."
        )
    return ds_session_id, authorization_token, cookies


class DeepSeekChat:
    def __init__(self, ds_session_id, authorization_token, cookies=None, device_id=None):
        self.base_url = "https://chat.deepseek.com"
        self.session = requests.Session()
        self.authorization = authorization_token
        self.device_id = device_id or os.environ.get("DS_DEVICE_ID") or str(uuid.uuid4())

        self._apply_cookies(ds_session_id, cookies)

        self.headers = {
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "authorization": self.authorization,
            "content-type": "application/json",
            "origin": self.base_url,
            "priority": "u=1, i",
            "sec-ch-ua": SEC_CH_UA,
            "sec-ch-ua-mobile": "?1",
            "sec-ch-ua-platform": '"Android"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": USER_AGENT,
            "x-client-bundle-id": CLIENT_BUNDLE_ID,
            "x-client-locale": CLIENT_LOCALE,
            "x-client-platform": "web",
            "x-client-timezone-offset": str(_tz_offset_seconds()),
            "x-client-version": CLIENT_VERSION,
            "x-device-id": self.device_id,
            "x-device-model": "",
        }
        self.chat_session_id = None
        self.parent_message_id = None

    # ------------------------------------------------------------------ cookies
    def _apply_cookies(self, ds_session_id, cookies):
        if cookies:
            # Raw "a=b; c=d" string or dict; wins over a bare ds_session_id.
            if isinstance(cookies, str):
                for chunk in cookies.split(";"):
                    if "=" in chunk:
                        name, _, value = chunk.strip().partition("=")
                        if name:
                            self.session.cookies.set(name, value)
            else:
                for name, value in cookies.items():
                    self.session.cookies.set(name, value)
            if ds_session_id and "ds_session_id" not in self.session.cookies:
                self.session.cookies.set("ds_session_id", ds_session_id)
        elif ds_session_id:
            self.session.cookies.set("ds_session_id", ds_session_id)

    # ----------------------------------------------------------------- helpers
    def _referer(self):
        if self.chat_session_id:
            return f"{self.base_url}/a/chat/s/{self.chat_session_id}"
        return f"{self.base_url}/"

    def _post_json(self, url, payload, extra_headers=None, timeout=30, retries=2):
        """POST a JSON body with bounded retries on transport errors."""
        headers = self.headers.copy()
        headers["Referer"] = self._referer()
        if extra_headers:
            headers.update(extra_headers)
        body = payload if isinstance(payload, str) else json.dumps(payload)
        last_error = None
        for attempt in range(retries + 1):
            try:
                return self.session.post(url, headers=headers, data=body, timeout=timeout)
            except requests.RequestException as error:
                last_error = error
                _warn(f"POST {url} failed (attempt {attempt + 1}/{retries + 1}): {error}")
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise last_error

    # ------------------------------------------------------------------- calls
    def create_chat_session(self):
        url = f"{self.base_url}/api/v0/chat_session/create"
        try:
            response = self._post_json(url, {})
        except requests.RequestException as error:
            _warn(f"create_chat_session transport error: {error}")
            return False
        if response.status_code != 200:
            _warn(f"create_chat_session HTTP {response.status_code}: {response.text[:200]}")
            return False
        result = response.json()
        if result.get("code") == 0:
            # Current shape is data.biz_data.chat_session.id; older builds used
            # data.biz_data.id. Accept either.
            holder = _first_present(result, "data.biz_data.chat_session", "data.biz_data")
            session_id = holder.get("id") if isinstance(holder, dict) else None
            if not session_id:
                _warn(f"create_chat_session: no session id in {json.dumps(result)[:200]}")
                return False
            self.chat_session_id = session_id
            return True
        _warn(f"create_chat_session rejected: {json.dumps(result)[:200]}")
        return False

    def create_pow_challenge(self, target_path=POW_TARGET_PATH):
        url = f"{self.base_url}/api/v0/chat/create_pow_challenge"
        try:
            response = self._post_json(url, {"target_path": target_path})
        except requests.RequestException as error:
            _warn(f"create_pow_challenge transport error: {error}")
            return None
        if response.status_code != 200:
            _warn(f"create_pow_challenge HTTP {response.status_code}: {response.text[:200]}")
            return None
        result = response.json()
        if result.get("code") == 0:
            challenge = _first_present(result, "data.biz_data.challenge",
                                       "data.biz_data.pow_challenge")
            if not challenge:
                _warn(f"create_pow_challenge: no challenge in {json.dumps(result)[:200]}")
                return None
            return challenge
        _warn(f"create_pow_challenge rejected: {json.dumps(result)[:200]}")
        return None

    def solve_pow_challenge(self, challenge_data, target_path=POW_TARGET_PATH):
        try:
            from .DeepSeekWASM import solve_wasm
        except ImportError:
            raise
        try:
            value, pow_response = solve_wasm(
                challenge_data["algorithm"],
                challenge_data["challenge"],
                challenge_data["salt"],
                challenge_data["expire_at"],
                challenge_data["difficulty"],
                challenge_data["signature"],
                challenge_data.get("target_path", target_path),
            )
            return bool(value), pow_response
        except Exception:
            _warn(backtrace())
            return False, ''

    # ----------------------------------------------------------------- message
    def send_message(self, message, printing=None, thinking_enabled=False,
                     search_enabled=False, model_type="default", on_delta=None):
        """Send `message` and return {"ok": bool, "content": ...}.

        printing   True -> stream to stdout; a callable -> used as the display
                   sink; falsy -> silent.
        on_delta   optional callable(kind, text) with kind in
                   {"think","response","tip","search","status"}; receives only
                   content, never the STREAMDOWN separator markers.
        """
        if model_type not in VALID_MODEL_TYPES:
            return {"ok": False,
                    "content": f"model_type must be one of {VALID_MODEL_TYPES!r}, got {model_type!r}."}
        if not self.chat_session_id and not self.create_chat_session():
            return {"ok": False, "content": "Can't create chat session."}

        challenge_data = self.create_pow_challenge()
        if not challenge_data:
            return {"ok": False, "content": "Can't create PoW challenge."}
        value, pow_response = self.solve_pow_challenge(challenge_data)
        if not value:
            return {"ok": False, "content": "Can't solve PoW challenge."}

        url = f"{self.base_url}/api/v0/chat/completion"
        payload = {
            "chat_session_id": self.chat_session_id,
            "parent_message_id": self.parent_message_id,
            "model_type": model_type,
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": bool(thinking_enabled),
            "search_enabled": bool(search_enabled),
            "action": None,
            "preempt": False,
        }

        def display(text):
            if callable(printing):
                printing(text)
            elif printing:
                print(text, end='', flush=True)

        def emit(kind, text):
            if on_delta:
                on_delta(kind, text)
            if kind != "tip":
                display(text)

        try:
            response = self.session.post(
                url,
                headers={**self.headers,
                         "Referer": self._referer(),
                         "x-ds-pow-response": pow_response},
                data=json.dumps(payload),
                timeout=180,
                stream=True,
            )
            if response.status_code != 200:
                body = response.text
                _warn(f"completion HTTP {response.status_code}: {body[:200]}")
                return {"ok": False, "content": f"HTTP {response.status_code}", "body": body}

            content_type = response.headers.get('content-type', '')
            if "text/event-stream" not in content_type:
                full_content = b''
                for chunk in response.iter_content(chunk_size=8192):
                    full_content += chunk
                # The service reports application errors as JSON on a 200, e.g.
                # {"code":40300,"msg":"MISSING_HEADER"}. Surface them legibly.
                try:
                    error_body = json.loads(full_content.decode('utf-8', 'replace'))
                except ValueError:
                    error_body = None
                if isinstance(error_body, dict) and error_body.get("code") not in (None, 0):
                    message = (f"DeepSeek error {error_body.get('code')}: "
                               f"{error_body.get('msg') or error_body}")
                    _warn(message)
                    return {"ok": False, "content": message, "error": error_body}
                _warn(f"non-SSE completion response: {full_content[:200]!r}")
                return {"ok": False, "content": full_content}

            state = self._parse_stream(response.iter_lines(decode_unicode=True),
                                       emit=emit, display=display,
                                       show_markers=bool(printing))
        except Exception:
            _warn(backtrace())
            return {"ok": False, "content": backtrace()}

        if state["msgid"]:
            self.parent_message_id = state["msgid"]

        if printing:
            print(f"\nFinished generating... Thinking time: {state['thinktime']} "
                  f"Total tokens: {state['tokencount']} "
                  f"{'Title: ' + state['title'] if state['title'] else ''}")

        ret = {
            "thinking_enabled": bool(thinking_enabled),
            "search_enabled": bool(search_enabled),
            "model_type": model_type,
            "response": state["respond"],
        }
        if thinking_enabled:
            ret["thinktime"] = state["thinktime"]
            ret["thought"] = state["think"]
        if search_enabled:
            ret["citation"] = state["citation"]
        if state["title"]:
            ret["title"] = state["title"]
        return {"ok": True, "content": ret}

    # -------------------------------------------------------------- stream parse
    def _parse_stream(self, lines, emit, display, show_markers=False):
        """Parse the SSE stream into a state dict.

        The stream is a JSON-patch protocol over a `response` object holding
        `fragments`, each fragment carrying a `type` (THINK / RESPONSE / SEARCH /
        TIP) and its own `content`:

            {"v":{"response":{...,"fragments":[{"id":2,"type":"THINK","content":"We"}]}}}
            {"p":"response/fragments/-1/content","o":"APPEND","v":" need"}
            {"v":" answer"}                                   <- bare continuation
            {"p":"response/fragments","o":"APPEND","v":[{"id":3,"type":"RESPONSE","content":"Hi"}]}
            {"p":"response/fragments/-1/elapsed_secs","o":"SET","v":"0.83"}
            {"p":"response","o":"BATCH","v":[{"p":"accumulated_token_usage","v":67}, ...]}
            {"p":"response/status","o":"SET","v":"FINISHED"}

        A `{"type": ...}` mode-switch protocol is also handled for older
        streams. Unknown events and fields are skipped, never fatal.
        """
        state = {
            "think": "", "respond": "", "generate_mode": "",
            "parid": None, "msgid": None, "reqid": None, "resid": None,
            "tokencount": None, "title": "", "thinktime": 0, "citation": {},
            "fragments": [], "target": None,
        }
        event = None

        # -- accumulation --------------------------------------------------
        def sink_for(fragment_type):
            return FRAGMENT_SINKS.get((fragment_type or "").upper(), "response")

        def route(fragment_type, text):
            sink = sink_for(fragment_type)
            if sink == "think":
                state["think"] += text
            elif sink == "response":
                state["respond"] += text
            emit(sink, text)

        def start_fragment(obj):
            fragment = {
                "type": (obj.get("type") or "RESPONSE").upper(),
                "content": obj.get("content") or "",
                "elapsed_secs": obj.get("elapsed_secs"),
            }
            state["fragments"].append(fragment)
            state["target"] = fragment
            state["generate_mode"] = fragment["type"]
            if show_markers:
                display(f"\n\n-----\nSTART {fragment['type']}\n")
            if fragment["content"]:
                route(fragment["type"], fragment["content"])
            return fragment

        def resolve_index(token):
            if token == '-1':
                return len(state["fragments"]) - 1
            try:
                return int(token)
            except (TypeError, ValueError):
                return None

        def fragment_at(token):
            index = resolve_index(token)
            if index is None or not (0 <= index < len(state["fragments"])):
                return None
            return state["fragments"][index]

        def append_text(text):
            if not isinstance(text, str):
                return
            if state["target"] is not None:
                state["target"]["content"] += text
                route(state["target"]["type"], text)
            elif state["generate_mode"]:
                route(state["generate_mode"], text)  # legacy `{"type": ...}` streams
            else:
                _warn(f"text outside of any fragment, dropped: {text[:60]!r}")

        # -- response object ------------------------------------------------
        def apply_response_object(obj):
            if not isinstance(obj, dict):
                return
            for key, field in (("message_id", "msgid"), ("parent_id", "parid")):
                if obj.get(key) is not None:
                    state[field] = obj[key]
            if obj.get("accumulated_token_usage") is not None:
                state["tokencount"] = obj["accumulated_token_usage"]
            if obj.get("status"):
                emit("status", str(obj["status"]))
            for fragment in obj.get("fragments") or []:
                if isinstance(fragment, dict):
                    start_fragment(fragment)

        def apply_response_field(field, value):
            if field in ("accumulated_token_usage",):
                state["tokencount"] = value
            elif field in ("elapsed_secs",):
                state["thinktime"] = _as_number(value)
            elif field == "status":
                emit("status", str(value))
            elif field in ("quasi_status", "has_pending_fragment", "conversation_mode",
                           "auto_continue", "search_triggered"):
                pass
            else:
                _warn(f"unknown response field {field!r}, skipped")

        # -- patches --------------------------------------------------------
        def apply_patch(data):
            path = data.get('p')
            op = data.get('o')
            value = data.get('v')

            if not path:
                # Bare continuation: applies to the last addressed fragment.
                append_text(value)
                return

            parts = str(path).strip('/').split('/')

            if parts[0] != 'response':
                handle_legacy(data, parts)
                return

            if len(parts) == 1:
                if op == 'BATCH' and isinstance(value, list):
                    for sub in value:
                        if isinstance(sub, dict):
                            apply_patch(sub)
                else:
                    _warn(f"unsupported response patch {op!r}, skipped")
                return

            if parts[1] == 'fragments':
                if len(parts) == 2:
                    items = value if isinstance(value, list) else [value]
                    for item in items:
                        if isinstance(item, dict):
                            start_fragment(item)
                    return
                fragment = fragment_at(parts[2])
                if fragment is None:
                    # No such fragment yet (missed the initial object, or an
                    # unexpected shape). Never drop content: fall back to the
                    # current target / mode.
                    field = parts[3] if len(parts) > 3 else 'content'
                    if field == 'content':
                        _warn(f"patch to unknown fragment {parts[2]!r}, using fallback")
                        append_text(value)
                    else:
                        _warn(f"patch to unknown fragment {parts[2]!r}, skipped")
                    return
                state["target"] = fragment
                field = parts[3] if len(parts) > 3 else 'content'
                if field == 'content':
                    if op == 'SET':
                        fragment['content'] = value if isinstance(value, str) else ''
                        route(fragment['type'], fragment['content'])
                    else:
                        append_text(value)
                else:
                    fragment[field] = value
                    if field == 'elapsed_secs':
                        state["thinktime"] = _as_number(value)
                return

            apply_response_field(parts[1], value)

        def handle_legacy(data, parts):
            """Older `{"type": ...}` mode-switch protocol."""
            tail = parts[-1] if parts else ''
            value = data.get('v')
            if tail in ('response', 'content', 'fragment', 'fragments'):
                append_text(value) if isinstance(value, str) else dispatch(value)
            elif tail == 'status':
                emit("status", str(value))
            elif tail == 'accumulated_token_usage':
                state["tokencount"] = value
            elif tail == 'elapsed_secs':
                state["thinktime"] = _as_number(value)
            elif tail in ('has_pending_fragment', 'conversation_mode', 'quasi_status',
                          'results'):
                if tail == 'results' and state["generate_mode"] == 'SEARCH':
                    dispatch(value)
            else:
                _warn(f"unknown path {data.get('p')!r}, skipped")

        # -- dispatch -------------------------------------------------------
        def dispatch(data):
            if not data or isinstance(data, bool):
                return
            if isinstance(data, str):
                if state["generate_mode"] == 'TIP':
                    emit("tip", data)
                else:
                    append_text(data)
                return
            if isinstance(data, list):
                for item in data:
                    dispatch(item)
                return
            if not isinstance(data, dict):
                _warn(f"unrecognisable payload type {type(data).__name__}, skipped")
                return

            # Search hits arrive as bare dicts carrying a url.
            if state["generate_mode"] == 'SEARCH' and 'url' in data:
                state["citation"][data.get('cite_index')] = data
                emit("search", f"{data.get('cite_index','?')}. "
                               f"[{data.get('title', data.get('site_name','UNKNOWN'))}"
                               f" - {data.get('site_name','UNKNOWN')}]({data['url']})\n")
                emit("search", '> ' + data.get('snippet', '') + "\n")
                return

            if set(data) == {'v'}:
                dispatch(data['v'])
                return
            if set(data) == {'response'}:
                apply_response_object(data['response'])
                return
            if set(data) <= {'message_id', 'parent_id'} and data:
                if data.get('message_id') is not None:
                    state["msgid"] = data['message_id']
                if data.get('parent_id') is not None:
                    state["parid"] = data['parent_id']
                return
            if 'p' in data:
                apply_patch(data)
                return
            if 'type' in data and data['type']:
                # Legacy: mode switch carried inline with content.
                state["generate_mode"] = str(data['type']).upper()
                if show_markers:
                    display(f"\n\n-----\nSTART {state['generate_mode']}\n")
                if 'content' in data:
                    dispatch(data['content'])
                return
            if set(data) == {'updated_at'}:
                return
            _warn(f"unrecognised dict payload, skipped: "
                  f"{json.dumps(data, separators=(',', ':'))[:120]}")

        # -- SSE framing ----------------------------------------------------
        display('\n')
        for line in lines:
            if not line:
                continue
            if not isinstance(line, str):
                line = line.decode('utf-8', 'replace')
            if line.startswith('data: '):
                raw = line[6:]
                if raw.strip() in ('', '[DONE]'):
                    continue
                try:
                    data = json.loads(raw)
                except ValueError:
                    _warn(f"non-JSON data line, skipped: {raw[:120]}")
                    continue
                if event in PAYLOAD_EVENTS:
                    dispatch(data)
                elif event == 'title':
                    if isinstance(data, dict) and 'content' in data:
                        state["title"] = data['content']
                elif event == 'ready':
                    if isinstance(data, dict):
                        state["reqid"] = data.get('request_message_id', state["reqid"])
                        state["resid"] = data.get('response_message_id', state["resid"])
                elif event in CLOSE_EVENTS:
                    pass
                else:
                    _warn(f"unknown event {event!r}, data skipped: {line[:120]}")
            elif line.startswith('event: '):
                event = line[7:].strip()
                if event in CLOSE_EVENTS:
                    break
            elif line.startswith(':'):
                continue  # SSE comment / heartbeat
            else:
                _warn(f"unrecognised stream line, skipped: {line[:120]}")
        return state
