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
from scipy.signal import lfilter, resample_poly

MODEL_ID = os.environ.get("KANI_MODEL_ID", "nineninesix/kani-tts-2-pt")
REFERENCE_AUDIO_PATH = os.environ.get("REFERENCE_AUDIO_PATH", "/app/reference_voice.wav")
NATIVE_SAMPLE_RATE = 22050  # см. model card kani-tts-2-pt
SPEED_FACTOR = float(os.environ.get("KANI_SPEED_FACTOR", "1.0"))  # ponytail: 1.15 звучал "пьяно" (resample_poly
# меняет тон/смазывает согласные, это не time-stretch) — выключено; включать заново только вместе с нормальным
# time-stretch (librosa/pyrubberband), а не голым ресемплингом

print("Loading KaniTTS-2 model (cold start)...")
model = KaniTTS(MODEL_ID, show_info=False)
embedder = SpeakerEmbedder(device="cuda")

print(f"Computing speaker embedding from {REFERENCE_AUDIO_PATH} ...")
speaker_embedding = embedder.embed_audio_file(REFERENCE_AUDIO_PATH)
print("Ready.")


def _normalize_loudness(
    audio_f32: np.ndarray,
    target_rms: float = 0.15,
    smooth_ms: float = 50.0,
    noise_floor: float = 0.02,
    max_gain: float = 3.0,
) -> np.ndarray:
    """AGC на непрерывной огибающей (однополюсный фильтр по мощности сигнала),
    без блочных "ступенек" усиления между окнами — прежняя блочная версия резко
    переключала gain каждые 200мс, что и звучало как "волны" громкости.
    noise_floor не даёт задирать усиление в тихих паузах/фоновом шуме."""
    n = len(audio_f32)
    if n == 0:
        return audio_f32
    power = audio_f32.astype(np.float64) ** 2
    alpha = float(np.exp(-1.0 / (smooth_ms / 1000 * NATIVE_SAMPLE_RATE)))
    envelope_power = lfilter([1 - alpha], [1, -alpha], power)
    rms_env = np.sqrt(np.maximum(envelope_power, 0.0)) + 1e-6
    gain = np.clip(target_rms / rms_env, 1.0 / max_gain, max_gain)
    gain = np.where(rms_env < noise_floor, 1.0, gain)
    out = (audio_f32 * gain).astype(np.float32)
    peak = float(np.max(np.abs(out))) + 1e-9
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def _adjust_speed(audio_f32: np.ndarray, factor: float) -> np.ndarray:
    """Ускоряет воспроизведение ресемплингом (сжимает длительность в factor раз).
    Побочный эффект — тон чуть выше при ускорении; для небольшого шага (~1.1-1.2x)
    почти незаметно на слух, отдельного time-stretch (без изменения тона) не заводим,
    чтобы не тащить лишнюю тяжёлую зависимость в и так хрупкую сборку Docker-образа."""
    if factor == 1.0 or len(audio_f32) == 0:
        return audio_f32
    down = max(1, int(round(1000 * factor)))
    return resample_poly(audio_f32, 1000, down)


def _to_pcm16(audio_f32: np.ndarray, src_rate: int, dst_rate: int) -> bytes:
    audio_f32 = _adjust_speed(audio_f32, SPEED_FACTOR)
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
