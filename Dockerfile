# RunPod Serverless образ для KaniTTS-2 (voice cloning, ky/ru/en)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git libsndfile1 sox libsox-dev libsox-fmt-all \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install numpy>=1.24.0 typing_extensions setuptools wheel \
    && pip3 install --no-build-isolation -r requirements.txt \
    && pip3 install --no-deps transformers==4.56.0 tokenizers==0.22.0

# Референс-голос владельца (склеен из voice_clone_ky/wav через prep_reference.py)
COPY reference_voice.wav /app/reference_voice.wav

# Прогреваем/скачиваем веса модели на этапе сборки образа, чтобы cold start
# serverless-запуска не тянул ~1-2GB весов из интернета каждый раз.
RUN python3 -c "from kani_tts import KaniTTS, SpeakerEmbedder; KaniTTS('nineninesix/kani-tts-2-pt', show_info=False); SpeakerEmbedder()"

COPY handler.py .

CMD ["python3", "-u", "handler.py"]
