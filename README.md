# Sesame Conversational Speech Model (CSM)

A self-hosted, real-time speech synthesis + conversation stack.
This project upgrades the original Sesame CSM into a **true streaming system**, enabling low-latency TTS and seamless LLM-driven conversational experiences.

The repo includes:

1. A streaming text-to-speech backend
2. An LLM + TTS pipeline
3. A conversational Maya AI wrapper
4. Tools for training / voice cloning
5. Demo clients + web integration

> Purpose: Gain full control of TTS performance, customization, and cost — without relying on Google TTS or other external providers.

---

## Table of Contents

1. Introduction
2. System Architecture
3. Project Structure
4. Deployment & Infrastructure
5. Integration
6. Cost Comparisons
7. Problem Summary & Scaling Solutions

---

# 1. Introduction

## 1.1 Overview

The Sesame CSM Streaming TTS project converts the existing Sesame CSM text-to-speech system
from a **blocking** process to a **streaming** one.
In the original system, the voice output was created only after the full text was processed.
This new version generates and streams audio **while** text is still being processed.

This enables:

-   Low latency
-   Real‑time interaction
-   Natural conversational flow

A web interface also exists where users can talk with an LLM and receive synchronous speech responses.

## 1.2 Purpose

The main goal is to create a **self‑hosted TTS** system fully integrated into HOOT, eliminating the
need for Google TTS or other third‑party providers.

Benefits:

-   Lower cost
-   Full control over performance
-   Custom voice support
-   Private infrastructure

## 1.3 Core Problems Solved

-   Converts blocking TTS → **streaming**
-   Low latency real‑time synthesis
-   Removes reliance on external providers
-   Allows custom voice integration
-   Conversational LLM-backed speech
-   Zero incremental cost beyond infra

## 1.4 Tech Stack

-   **Python + FastAPI**
-   **AWS EC2**
-   **Cloudflare Tunnel**
-   **PM2**

---

# 2. System Architecture

## 2.1 High-Level Diagram

```
Frontend (Web/App)
        │
        ▼
     WebSocket
        │
        ▼
Backend Server
        │
        ▼
Sesame TTS Model
        │
        ▼
Audio Chunks → Frontend Playback
```

## 2.2 Data Flow Summary

```
User types text
    ↓
Frontend sends text + voice_id
    ↓
Backend streams chunked audio
    ↓
Frontend plays audio in near realtime
```

## 2.3 Low-Level Flow

1. Frontend sends:

    - text
    - voice_id
      via WebSocket

2. Backend locks TTS model

    - Only one active request at a time

3. Model streams audio chunks
   (NumPy float32 arrays)

4. Backend pushes messages:

    - `audio_chunk`
    - `audio_status`
    - `completion`
    - `error`

5. Frontend reconstructs & plays audio

6. Disconnect cleanup

---

# 3. Project Structure

```
csm-complete/
├── csm-basic/
├── csm-app/
├── csm-client/
│   ├── newchunks.py
│   └── llm.py
├── csm-stream/
│   ├── config.json
│   ├── hive-tts.py
│   ├── main.py
│   ├── tera.py
│   ├── lora.py
│   ├── setup.py
│   ├── audio_data/
│   └── models/
├── goop/
└── voices/
```

## Folder Explanations

### `csm-basic/`

Original blocking CSM model implementation.

### `csm-app/`

Early mobile prototype for streaming.

### `csm-client/`

Terminal-based client tools

-   `newchunks.py` → Stream TTS
-   `llm.py` → Chat with LLM backend

### `csm-stream/`

Core backend + training tools:

-   `hive-tts.py` → Streaming TTS server
-   `tera.py` → LLM + TTS layer
-   `main.py` → Conversational AI / Maya AI
-   `lora.py` → Finetune/clone voices
-   `setup.py` → Install dependencies

Contains:

-   `audio_data/`
-   `models/`

### `goop/`

Web demo using WebChannelSocket + FlutterSound.

### `voices/`

Stores pre-defined voice references.

---

# 4. Deployment & Infrastructure

## 4.1 Docker

Base image:

```
nvcr.io/nvidia/pytorch:24.08-py3
```

Run:

```
docker run --gpus all -it --rm -p 8888:8888 nvcr.io/nvidia/pytorch:24.08-py3
```

Optional:

```
jupyter notebook ...
```

## 4.2 AWS Deployment

-   Instance: `g6.xlarge`
-   GPU: NVIDIA L4 24GB
-   VRAM usage: ~4–5GB

Setup:

```
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535 nvidia-cuda-toolkit
sudo reboot
```

Install Python:

```
sudo apt install -y python3.10 python3.10-venv ...
```

Project setup:

```
git clone <repo>
cd <repo>
python3.10 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Model download:

```
huggingface-cli download
```

## 4.3 Cloudflare

Create tunnel → expose backend securely.

## 4.4 PM2

Run:

```
pm2 start hive-tts.py
```

---

# 5. Integration

## 5.1 HOOT App

Replaced Google TTS with Sesame Streaming TTS.

Uses WebSocket streaming.

## 5.2 AI Conversation

Flow:

```
Mic → STT → LLM → TTS → streaming audio
```

Run:

```
python setup.py
python main.py
```

---

# 6. Cost Comparison

| Users | GPUs Needed | Cost/hr |
| ----- | ----------- | ------- |
| 1     | 1           | ₹79     |
| 50    | 13          | ₹1030   |

Current model loads 1 per user → expensive.

---

# 7. Problem Summary & Scaling

### Issues

-   1 running model per user
-   High VRAM consumption (5GB/user)
