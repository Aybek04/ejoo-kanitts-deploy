"""Склеивает несколько WAV-клипов голоса владельца в один референс-файл
для voice cloning (rekомендация KaniTTS-2: 10-20 сек, чистый голос).

Использование:
    python prep_reference.py <папка_с_wav> <выходной.wav> [макс_секунд]

Берёт файлы по порядку, пока не наберёт max_seconds (по умолчанию 15).
Только stdlib (wave) — все входные файлы уже моно 44.1кГц.
"""
import sys
import wave
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    max_seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

    wavs = sorted(src_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Нет .wav файлов в {src_dir}")

    with wave.open(str(wavs[0]), "rb") as first:
        params = first.getparams()

    total_frames_needed = int(max_seconds * params.framerate)
    frames_written = 0

    with wave.open(str(out_path), "wb") as out:
        out.setparams(params)
        for wav_path in wavs:
            if frames_written >= total_frames_needed:
                break
            with wave.open(str(wav_path), "rb") as src:
                if src.getparams()[:3] != params[:3]:
                    print(f"skip (формат не совпадает): {wav_path.name}")
                    continue
                remaining = total_frames_needed - frames_written
                frames = src.readframes(min(src.getnframes(), remaining))
                out.writeframes(frames)
                frames_written += src.getnframes()

    print(f"Готово: {out_path} (~{frames_written / params.framerate:.1f} сек)")


if __name__ == "__main__":
    main()
