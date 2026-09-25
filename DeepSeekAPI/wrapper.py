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

from enum import Enum
import os
import subprocess
import sys

from .DeepSeekChat import DeepSeekChat


class IOMethods(Enum):
    PRINT = 1
    RETURN = 2
    STREAMDOWN = 3


def _cookie_string(cookies):
    if not cookies:
        return None
    if isinstance(cookies, str):
        return cookies
    return "; ".join(f"{name}={value}" for name, value in cookies.items())


def DeepSeekChatExample(DS_SESSION_ID, AUTHORIZATION_TOKEN, message, mode,
                        thinking_enabled=False, search_enabled=False,
                        model_type="default", cookies=None, device_id=None):
    if mode == IOMethods.STREAMDOWN:
        # The renderer runs in a child process. Credentials travel via the
        # environment, never via argv, so they do not show up in `ps`.
        env = dict(os.environ, DS_SESSION_ID=DS_SESSION_ID,
                   AUTHORIZATION_TOKEN=AUTHORIZATION_TOKEN)
        cookie_string = _cookie_string(cookies)
        if cookie_string:
            env["DS_COOKIES"] = cookie_string
        if device_id:
            env["DS_DEVICE_ID"] = device_id
        program = (
            "from DeepSeekAPI import DeepSeekChat;"
            "from DeepSeekAPI.DeepSeekChat.main import load_tokens;"
            "sid,tok,ck=load_tokens();"
            f"DeepSeekChat(sid,tok,cookies=ck).send_message({message!r},True,"
            f"{thinking_enabled!r},{search_enabled!r},{model_type!r})"
        )
        proc = subprocess.Popen(
            [sys.executable, '-c', program],
            stdout=subprocess.PIPE,
            text=False,
            env=env,
        )
        from .streamdown import init, emit
        init()
        emit(proc.stdout)
        return
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN, cookies=cookies,
                        device_id=device_id)
    ret = chat.send_message(message, mode == IOMethods.PRINT,
                            thinking_enabled, search_enabled, model_type)
    if mode == IOMethods.PRINT:
        return
    return ret
