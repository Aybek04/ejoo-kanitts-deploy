# KaniTTS-2 voice cloning → RunPod Serverless → Vapi Custom TTS

Клонирование голоса владельца (кыргызский/русский/английский) через
open-source модель **KaniTTS-2**, задеплоенную как **RunPod Serverless**
(масштабирование до нуля — платим только за реальный инференс, без
постоянно включённого GPU).

## Что подтверждено разведкой (актуально на момент подготовки)

- Модель реальна и актуальна: **`nineninesix/kani-tts-2-pt`**
  (Hugging Face), 400M параметров, языки **en / es / ky** — кыргызский
  официально заявлен в теге модели.
  https://huggingface.co/nineninesix/kani-tts-2-pt
- Библиотека: **`kani-tts-2`** (PyPI/GitHub `nineninesix-ai/kani-tts-2`),
  требует `transformers==4.56.0`. Лицензия пакета — Apache-2.0/MIT, лицензия
  **весов модели** — `lfm1.0` (Liquid AI license, т.к. бэкбон LFM2) —
  см. https://www.liquid.ai/lfm-license перед коммерческим использованием.
- Voice cloning — **zero-shot через speaker embedding**: `SpeakerEmbedder`
  (WavLM-based) извлекает 128-мерный эмбеддинг из референс-WAV (рекомендуют
  10-20 сек чистой речи), эмбеддинг передаётся в `model(text, speaker_emb=...)`.
  Дообучение не нужно.
- VRAM: **~3GB** (замер разработчиков на RTX 5080), совпадает с тем, что
  Айбек уже знал — вписывается в дешёвые serverless GPU тиры RunPod.
  Нативная частота дискретизации аудио на выходе — **22050 Hz**.
  Максимальная длина генерации — до ~40 сек за один вызов.
- Референс-WAV для клонирования уже готовы:
  `C:\Users\user\Desktop\Claude\ejoo\Телефония\voice_clone_ky\wav\` (24 файла).

## Формат Vapi Custom TTS (подтверждено из офиц. доков)

Vapi шлёт **POST** на ваш `voice.server.url` с телом:
```json
{
  "message": {
    "type": "voice-request",
    "text": "...",
    "sampleRate": 24000,
    "timestamp": 1234567890
  }
}
```
`sampleRate` — одно из `8000 / 16000 / 22050 / 24000`.

Ваш ответ обязан быть:
- HTTP 200
- `Content-Type: application/octet-stream`
- **сырые PCM-байты в теле ответа напрямую** (без JSON, без base64, без
  WAV-заголовка): mono, 16-bit signed little-endian, ровно на той частоте,
  что попросили в `sampleRate`.

Аутентификация — заголовок `X-VAPI-SECRET` (или `credentialId` через Custom
Credentials в дашборде Vapi).

**Важный нюанс архитектуры:** RunPod Serverless API (`/runsync`) — это
JSON-in/JSON-out контракт (base64 в поле `output`), а не "сырой поток
по HTTP", которого ждёт Vapi. Поэтому напрямую RunPod endpoint как
`voice.server.url` не подключить — нужен небольшой прокси-слой
(`proxy_server.py` в этой папке), который лежит на своём постоянном
HTTPS-хосте, конвертирует JSON/base64 → raw PCM и отдаёт его Vapi.

## Файлы в этой папке

| Файл | Назначение |
|---|---|
| `Dockerfile` | образ для RunPod Serverless (CUDA 12.4 + kani-tts-2, веса модели скачиваются на этапе сборки) |
| `handler.py` | RunPod serverless handler: текст → speaker-cloned аудио → base64 PCM16 в JSON |
| `requirements.txt` | зависимости Python |
| `prep_reference.py` | склеивает несколько WAV из `voice_clone_ky\wav\` в один референс-файл 10-20 сек |
| `proxy_server.py` | мост Vapi (raw PCM webhook) ↔ RunPod (JSON/base64), только stdlib |

## Шаги деплоя на RunPod (когда будет аккаунт и ключи)

1. **Подготовить референс-голос** (локально, не требует GPU):
   ```
   python prep_reference.py "C:\Users\user\Desktop\Claude\ejoo\Телефония\voice_clone_ky\wav" reference_voice.wav 15
   ```
   Положить получившийся `reference_voice.wav` рядом с `Dockerfile` (он
   копируется в образ на этапе сборки).

2. **Собрать Docker-образ:**
   ```
   docker build -t kani-tts-ejoo:latest .
   ```

3. **Запушить в registry**, который поддерживает RunPod (Docker Hub или
   ghcr.io):
   ```
   docker tag kani-tts-ejoo:latest <ваш-registry>/kani-tts-ejoo:latest
   docker push <ваш-registry>/kani-tts-ejoo:latest
   ```

4. **Создать RunPod Serverless Endpoint** через дашборд
   https://www.runpod.io/console/serverless — указать образ из шага 3,
   GPU-тир с ~8-16GB VRAM (запас над требуемыми 3GB), `Min Workers = 0`
   (это и даёт масштабирование до нуля / оплату по факту).

5. **Запустить `proxy_server.py`** на постоянном хосте (подходит уже
   существующий VDS Hetzner из проекта ejoo — см. память "ejoo server"),
   передав в окружение `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`,
   `VAPI_WEBHOOK_SECRET`.

6. **Подключить в Vapi**: в конфиге ассистента —
   ```json
   {
     "voice": {
       "provider": "custom-voice",
       "server": {
         "url": "https://<хост-с-proxy_server.py>/synthesize",
         "secret": "<VAPI_WEBHOOK_SECRET>",
         "timeoutSeconds": 30
       }
     }
   }
   ```

## Что потребуется из секретов дальше (НЕ трогать здесь — только через secret-keeper)

- **RunPod API key** — для создания/управления serverless endpoint и для
  `RUNPOD_API_KEY` в `proxy_server.py`.
- **Docker registry credentials** (Docker Hub token или GHCR PAT) — для
  `docker push` образа.
- **`VAPI_WEBHOOK_SECRET`** — придумываемое самим значение (не берётся
  извне), но хранить его нужно тоже через хранилище секретов, не в коде.
- **Vapi API key** — для привязки `voice.server.url` к ассистенту через
  API/дашборд Vapi (или это можно сделать вручную в дашборде без API-ключа
  в коде).

Ничего из этого в текущей задаче не запрашивалось и не использовалось.

## Открытые вопросы / TODO

- Холодный старт RunPod Serverless (десятки секунд на первый запрос после
  простоя) может не укладываться в `timeoutSeconds` Vapi при низком трафике
  звонков — стоит проверить `Active Workers = 1` (не даёт скейлиться в 0,
  но убирает cold start) как компромисс, если задержка окажется критичной.
- `language_tag` для `ky` в `kani-tts-2-pt` — в коде оставлен опциональным
  параметром (`handler.py`), но точный список тегов модели надо проверить
  через `model.show_language_tags()` при первом реальном запуске (в этой
  задаче GPU не поднимался, чтобы не тратить деньги без аккаунта).
- Прокси (`proxy_server.py`) — простая однопоточная реализация без ретраев/
  очереди; если объём звонков вырастет, лазейка попроще на будущее —
  заменить на управляемый always-on сервис (Cloudflare Worker / Fly.io) вместо
  ручного VDS-процесса.
