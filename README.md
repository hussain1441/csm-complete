# Sesame Conversational Speech Model (CSM) — Streaming TTS

A simplified guide explaining how to install, set up, and use the four available
TTS server modes and their associated clients.

## ✅ 1) Setup

Before running ANY server or client:

### Install dependencies

```
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### Run model setup

```
python setup.py
```

This downloads / prepares TTS models so all servers can run.

## ✅ 2) Server Types

There are **4 TTS modes**, each with a matching simple client.
All servers live in `csm-stream/`.

```
csm-stream/
├── tts-server.py
├── tts-ai-server.py
├── tts-selection-server.py
├── tts-voice-server.py
└── setup.py
```

## ✅ A) Base Streaming TTS

File: `tts-server.py`
Client: `tts-client.py`

Description:
Streams synthesized speech audio while text is still being processed.

### Run Server

```
python tts-server.py
```

### Run Client

```
python tts-client.py
```

This sends text to the server, receives audio chunks, and plays audio.

## ✅ B) Conversational AI + TTS

File: `tts-ai-server.py`
Client: `tts-ai-client.py`

Description:
Use an LLM to generate text responses, then speak them aloud continuously
inside the command line.

### Run Server

```
python tts-ai-server.py
```

### Run Client

```
python tts-ai-client.py
```

You type → AI responds → Audio streams to you.

## ✅ C) Voice Selection TTS

File: `tts-selection-server.py`
Client: `tts-selection-client.py`
Config: `config_selection.json`

Description:
Allows choosing between multiple voices.
Uses config file to reference available voice models.

### Run Server

```
python tts-selection-server.py
```

### Run Client

```
python tts-selection-client.py
```

---

## ✅ D) Browser-Based Voice Chat

File: `tts-voice-server.py`

Description:
Starts a local web UI.
You can speak with it via browser; it streams TTS output back.

### Run Server

```
python tts-voice-server.py
```

Then open browser → link printed in terminal

No separate Python client needed.

## ✅ 3) Project Structure (Short)

```
csm-complete
├── csm_app/              # Early mobile UI
├── csm-client/           # TTS + AI sample clients
├── csm-stream/           # Core TTS + model servers
│   ├── audio_data/
│   ├── voices/
│   ├── config.json
│   ├── config_selection.json
│   ├── tts-server.py
│   ├── tts-ai-server.py
│   ├── tts-selection-server.py
│   ├── tts-voice-server.py
│   └── setup.py
└── hoot-implementation/
    └── hoot-tts-server.dart	# Hoot Implementation of TTS Streaming
```

## ✅ 4) Notes

-   All servers require running `setup.py` first
-   All streaming servers use WebSockets internally
-   Clients are minimal Python examples
-   `tts-voice-server.py` is browser‑based only

## ✅ 5) Quick Reference

| Mode            | Server                  | Client                  | UI  | Voices   |
| --------------- | ----------------------- | ----------------------- | --- | -------- |
| Base TTS        | tts-server.py           | tts-client.py           | No  | Single   |
| TTS + AI        | tts-ai-server.py        | tts-ai-client.py        | No  | Single   |
| Voice Selection | tts-selection-server.py | tts-selection-client.py | No  | Multiple |
| Web Chat        | tts-voice-server.py     | Browser                 | Yes | Multiple |
