"""Мост между Vapi Custom TTS webhook и RunPod Serverless endpoint.

Зачем нужен: Vapi шлёт POST {"message": {"type": "voice-request", "text",
"sampleRate", ...}} и ждёт HTTP 200 + Content-Type: application/octet-stream
+ сырые PCM16 mono LE байты В ТЕЛЕ ОТВЕТА НАПРЯМУЮ.
RunPod Serverless (/runsync) — это отдельный контракт: JSON-запрос,
JSON-ответ с base64 в поле output. Форматы несовместимы напрямую, поэтому
нужен этот прокси-слой.

Держать этот файл нужно на ЛЮБОМ постоянно доступном по HTTPS хосте
(например тот же VDS Hetzner, где уже крутится ejoo-телефония — см. память
"ejoo server"). RunPod serverless сам по себе не даёт стабильный "raw PCM"
HTTP-эндпоинт.

ENV (задаются на хосте, НЕ хардкодить, НЕ коммитить):
    RUNPOD_ENDPOINT_ID   — id serverless endpoint в RunPod
    RUNPOD_API_KEY       — API-ключ RunPod (через secret-keeper)
    VAPI_WEBHOOK_SECRET  — значение, которое Vapi шлёт в заголовке X-VAPI-SECRET
                            (сверяем, чтобы не дёргал кто попало)

Только stdlib (http.server + urllib) — лишних зависимостей не тянем.
"""
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlrequest
from urllib.error import HTTPError

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
VAPI_WEBHOOK_SECRET = os.environ.get("VAPI_WEBHOOK_SECRET")  # опционально, но рекомендуется
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"
PORT = int(os.environ.get("PORT", 8090))


def call_runpod(text: str, sample_rate: int) -> bytes:
    payload = json.dumps({"input": {"text": text, "sampleRate": sample_rate}}).encode()
    req = urlrequest.Request(
        RUNPOD_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=45) as resp:
        body = json.loads(resp.read())
    output = body.get("output") or {}
    if "error" in output:
        raise RuntimeError(output["error"])
    return base64.b64decode(output["audio_base64"])


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if VAPI_WEBHOOK_SECRET and self.headers.get("X-VAPI-SECRET") != VAPI_WEBHOOK_SECRET:
            self.send_response(401)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            message = payload["message"]
            text = message["text"]
            sample_rate = int(message["sampleRate"])
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(exc).encode())
            return

        try:
            pcm_bytes = call_runpod(text, sample_rate)
        except (HTTPError, RuntimeError, KeyError) as exc:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(exc).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(pcm_bytes)))
        self.end_headers()
        self.wfile.write(pcm_bytes)

    def log_message(self, fmt, *args):
        print(f"[proxy] {self.address_string()} - {fmt % args}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Vapi<->RunPod TTS proxy listening on :{PORT}")
    server.serve_forever()
