"""A stand-in for llama.cpp's ``llama-server``: just enough of its HTTP surface for the engine's tests.

Run as ``python fake_llama.py -m <model> --port <n> --host 127.0.0.1 …`` (the flags llm.Runtime passes).
Serves ``/health`` and ``/v1/chat/completions``. Environment variables pick a behaviour:

    FAKE_LLAMA_MODE=exit      exit at once (a server that will not start)
    FAKE_LLAMA_MODE=hang      accept the connection and never answer a completion
    FAKE_LLAMA_REPLY=<text>   the message content to return (default ``{"ok": true}``)
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

mode = os.environ.get("FAKE_LLAMA_MODE", "")
if mode == "exit":
    sys.exit(3)

port = int(sys.argv[sys.argv.index("--port") + 1])
reply = os.environ.get("FAKE_LLAMA_REPLY", '{"ok": true}')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, body: dict, status: int = 200):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        else:
            self._send({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self._send({}, 404)
            return
        if mode == "hang":
            time.sleep(60)
        # Echo what the engine asked for, so tests can assert on the request it built.
        self._send({
            "choices": [{"message": {"content": reply}}],
            "echo": {
                "json_mode": request.get("response_format"),
                "thinking": request.get("chat_template_kwargs"),
                "prompt": request["messages"][0]["content"][:200],
            },
        })


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
