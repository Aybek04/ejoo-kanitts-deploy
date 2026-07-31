"""Тестовый прогон KaniTTS на RunPod Serverless: 5 коротких кыргызских фраз.

НЕ ЗАПУСКАТЬ без RUNPOD_API_KEY и RUNPOD_ENDPOINT_ID в окружении —
оба секрета не должны попадать в код, только через переменные окружения /
secret-keeper.

Запуск (когда секреты будут доступны, например через sops-обёртку):
    RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... python test_inference.py

Каждая фраза уходит отдельным /runsync запросом (синхронный, ждёт ответа —
годится для разового прослушивания, для прод-нагрузки нужен /run + poll).
Результаты сохраняются как test_out_1.wav .. test_out_5.wav рядом со скриптом.
"""
import base64
import os
import sys
import wave

import requests

PHRASES = [
    "Салам, кандайсыз?",
    "Бул кыргыз тилинде сүйлөгөн жасалма үн.",
    "Ырахмат, жакшы күн болсун.",
    "Ассалому алейкум, кош келиңиз.",
    "Бул кыска сынак сүйлөм, беш инчи.",
]

SAMPLE_RATE = 22050  # нативная частота модели, без ресемплинга


def synthesize(api_key: str, endpoint_id: str, text: str) -> bytes:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": {"text": text, "sampleRate": SAMPLE_RATE}}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    output = data.get("output") or {}
    if "error" in output:
        raise RuntimeError(f"handler error: {output['error']}")
    if "audio_base64" not in output:
        raise RuntimeError(f"unexpected response: {data}")
    return base64.b64decode(output["audio_base64"])


def save_wav(path: str, pcm_bytes: bytes, sample_rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def main() -> None:
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        print(
            "STOP: нужны переменные окружения RUNPOD_API_KEY и RUNPOD_ENDPOINT_ID.\n"
            "Это секреты — брать их через secret-keeper, не хардкодить здесь.",
            file=sys.stderr,
        )
        sys.exit(1)

    for i, phrase in enumerate(PHRASES, start=1):
        print(f"[{i}/{len(PHRASES)}] {phrase!r} ...")
        pcm = synthesize(api_key, endpoint_id, phrase)
        out_path = f"test_out_{i}.wav"
        save_wav(out_path, pcm, SAMPLE_RATE)
        print(f"  -> {out_path} ({len(pcm)} bytes PCM)")

    print("Готово. Прослушать test_out_1.wav .. test_out_5.wav")


if __name__ == "__main__":
    main()
