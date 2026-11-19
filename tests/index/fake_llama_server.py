"""A stand-in for llama-server in the tests: answers /health and /props on 127.0.0.1.

    python fake_llama_server.py -m MODEL ... --port N

FAKE_LLAMA_MODE=hang answers /health with 503 forever, FAKE_LLAMA_MODE=exit quits at once.
FAKE_LLAMA_ENV_OUT=<file> writes the names of the environment variables it got to that file.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

args = sys.argv[1:]
model = args[args.index("-m") + 1]
port = int(args[args.index("--port") + 1])
mode = os.environ.get("FAKE_LLAMA_MODE", "")
if os.environ.get("FAKE_LLAMA_ENV_OUT"):
    with open(os.environ["FAKE_LLAMA_ENV_OUT"], "w", encoding="utf-8") as f:
        json.dump(sorted(os.environ), f)
if mode == "exit":
    sys.exit(3)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            code, body = (503, {"status": "loading"}) if mode == "hang" else (200, {"status": "ok"})
        elif self.path == "/props":
            code, body = 200, {"model_path": model}
        else:
            code, body = 404, {}
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", port), Handler).serve_forever()
