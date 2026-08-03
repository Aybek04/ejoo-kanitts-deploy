"""RunPod Serverless handler для KaniTTS-2 (voice cloning, кыргызский/русский/английский).

Вход (event["input"]):
    text          str   — текст для озвучки (обязательно)
    sampleRate    int   — целевая частота дискретизации, как требует Vapi
                          Custom TTS (8000/16000/22050/24000). По умолчанию 22050
                          (нативная частота модели — без ресемплинга).
    language_tag  str   — опционально, если модель поддерживает языковые теги
                          (проверить через model.show_language_tags() при первом запуске).

Выход (JSON, т.к. RunPod serverless всегда отдаёт JSON, не сырой поток):
    audio_base64  str — PCM16 mono little-endian, base64
    sample_rate   int
    format        str — "pcm_s16le_mono"

ВАЖНО: RunPod serverless API возвращает JSON (base64), а Vapi Custom TTS
ожидает HTTP 200 + Content-Type: application/octet-stream + сырые PCM-байты
в теле ответа напрямую. Между Vapi и этим handler'ом нужен небольшой прокси
(см. README.md, раздел "Прокси-мост Vapi <-> RunPod").
"""
import base64
import os

import numpy as np
import runpod
from kani_tts import KaniTTS, SpeakerEmbedder
from scipy.signal import resample_poly

MODEL_ID = os.environ.get("KANI_MODEL_ID", "nineninesix/kani-tts-2-pt")
REFERENCE_AUDIO_PATH = os.environ.get("REFERENCE_AUDIO_PATH", "/app/reference_voice.wav")
NATIVE_SAMPLE_RATE = 22050  # см. model card kani-tts-2-pt

print("Loading KaniTTS-2 model (cold start)...")
model = KaniTTS(MODEL_ID, show_info=False)
embedder = SpeakerEmbedder(device="cuda")

print(f"Computing speaker embedding from {REFERENCE_AUDIO_PATH} ...")
speaker_embedding = embedder.embed_audio_file(REFERENCE_AUDIO_PATH)
print("Ready.")


def _normalize_loudness(audio_f32: np.ndarray, window_ms: float = 200, target_rms: float = 0.15) -> np.ndarray:
    """Выравнивает "волны" громкости внутри фразы короткооконным AGC:
    считает RMS по окнам, сглаживает огибающую и приводит её к target_rms,
    ограничивая максимальное усиление, чтобы не поднимать шум в тихих паузах."""
    win_samples = max(1, int(window_ms / 1000 * 22050))
    n = len(audio_f32)
    if n == 0:
        return audio_f32
    envelope = np.zeros(n, dtype=np.float32)
    for start in range(0, n, win_samples):
        end = min(start + win_samples, n)
        rms = float(np.sqrt(np.mean(audio_f32[start:end] ** 2)) + 1e-6)
        envelope[start:end] = rms
    # сглаживаем огибающую, чтобы избежать резких скачков усиления между окнами
    smooth = np.convolve(envelope, np.ones(5) / 5, mode="same")
    gain = np.clip(target_rms / smooth, 0.5, 4.0)
    return audio_f32 * gain


def _to_pcm16(audio_f32: np.ndarray, src_rate: int, dst_rate: int) -> bytes:
    audio_f32 = _normalize_loudness(audio_f32)
    if src_rate != dst_rate:
        audio_f32 = resample_poly(audio_f32, dst_rate, src_rate)
    audio_f32 = np.clip(audio_f32, -1.0, 1.0)
    return (audio_f32 * 32767.0).astype("<i2").tobytes()


def handler(event):
    inp = event.get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    sample_rate = int(inp.get("sampleRate", NATIVE_SAMPLE_RATE))
    language_tag = inp.get("language_tag")

    if not text:
        return {"error": "empty text"}
    if sample_rate not in (8000, 16000, 22050, 24000):
        return {"error": f"unsupported sampleRate: {sample_rate}"}

    gen_kwargs = {"speaker_emb": speaker_embedding}
    if language_tag:
        gen_kwargs["language_tag"] = language_tag

    audio, _ = model(text, **gen_kwargs)  # float32 @ NATIVE_SAMPLE_RATE
    pcm_bytes = _to_pcm16(audio, NATIVE_SAMPLE_RATE, sample_rate)

    return {
        "audio_base64": base64.b64encode(pcm_bytes).decode("utf-8"),
        "sample_rate": sample_rate,
        "format": "pcm_s16le_mono",
    }


runpod.serverless.start({"handler": handler})
