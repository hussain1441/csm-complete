# works with tts-client.py
# orignal streaming file

import os

os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "CUDA_LAUNCH_BLOCKING": "1",
        "PYTORCH_DISABLE_CUDA_GRAPHS": "1",
    }
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import threading
import time
import json
import queue
import torch
import torchaudio
import re
import numpy as np
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from typing import Optional
from generator import load_csm_1b_local, Segment
import warnings

warnings.filterwarnings("ignore")

# ----- Globals -----
config = None
generator = None
model_ready = threading.Event()
model_thread_running = threading.Event()
model_queue = queue.Queue()
model_result_queue = queue.Queue()
audio_gen_lock = threading.Lock()
is_speaking = False
active_connections = []
reference_segments = []

# ----- FastAPI -----
app = FastAPI()


# ----- Config Model -----
class Config(BaseModel):
    model_path: str
    voice_speaker_id: int = 0
    reference_audio: str
    reference_text: str
    reference_audio2: Optional[str] = None
    reference_text2: Optional[str] = None
    reference_audio3: Optional[str] = None
    reference_text3: Optional[str] = None


# ----- Server Startup -----
@app.on_event("startup")
async def startup_event():
    global config, model_thread_running

    print("[Startup] Loading configuration...", flush=True)

    # Load config from file
    with open("config.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    config = Config(**data)

    print("[Startup] Configuration loaded", flush=True)
    print(f"Model path: {config.model_path}")

    # Load reference audio
    load_reference_segments()  # CHECK IT OUT

    # Start model worker thread
    model_thread_running.set()
    threading.Thread(target=model_worker, daemon=True, name="model_worker").start()

    print("[Startup] Server ready!", flush=True)


# ----- Load Reference Segments -----
def load_reference_segments():
    global reference_segments
    reference_segments = []

    # load primary reference
    primary_segment = load_single_reference(
        config.reference_audio, config.reference_text, config.voice_speaker_id
    )
    if not primary_segment:
        print("Error: Primary reference audio is required but not found!", flush=True)
        return

    reference_segments.append(primary_segment)

    # load second reference
    if config.reference_audio2 and config.reference_text2:
        segment = load_single_reference(
            config.reference_audio2, config.reference_text2, config.voice_speaker_id
        )
        if segment:
            reference_segments.append(segment)
        else:
            print("Second reference audio is not found!", flush=True)

    # load third reference
    if config.reference_audio3 and config.reference_text3:
        segment = load_single_reference(
            config.reference_audio3, config.reference_text3, config.voice_speaker_id
        )
        if segment:
            reference_segments.append(segment)
        else:
            print("Third reference audio is not found!", flush=True)


# ----- Helper Function To Load Single Reference Audio -----
def load_single_reference(audio_path, text, speaker_id):
    if os.path.isfile(audio_path):
        print(f"Loading reference audio: {audio_path}", flush=True)
        wav, sr = torchaudio.load(audio_path)
        wav = torchaudio.functional.resample(
            wav.squeeze(0), orig_freq=sr, new_freq=24000
        )
        return Segment(text=text, speaker=speaker_id, audio=wav)
    else:
        print(
            f"Warning: Reference audio '{config.reference_audio}' not found", flush=True
        )
        return None


# ----- Model Worker Thread -----
def model_worker():
    global generator

    print("[Worker] Starting model loading...", flush=True)

    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.fx_graph_cache = False

    # Load the voice model
    generator = load_csm_1b_local(config.model_path, "cuda")

    print("[Worker] CSM Model loaded. Starting warm-up...", flush=True)

    # Warm-up the model - TRY VARIATION
    warmup_text = "warm-up " * 5
    for chunk in generator.generate_stream(
        text=warmup_text,
        speaker=config.voice_speaker_id,
        context=reference_segments,
        max_audio_length_ms=1000,
        temperature=0.7,
        topk=40,
    ):
        pass

    # SET MODEL.READY
    model_ready.set()

    print("[Worker] Model warm-up complete!", flush=True)

    # Main worker loop - MAIN CODE
    while model_thread_running.is_set():
        try:
            request = model_queue.get(timeout=0.1)
            if request is None:
                break

            # data unpacking
            text, speaker_id, context, max_ms, temperature, topk = request

            # Generate audio stream
            for chunk in generator.generate_stream(
                text=text,
                speaker=speaker_id,
                context=context,
                max_audio_length_ms=max_ms,
                temperature=temperature,
                topk=topk,
            ):
                model_result_queue.put(chunk)
                if not model_thread_running.is_set():
                    break

            # Marker for audio end
            model_result_queue.put("EOS_AUDIO")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] Error: {e}", flush=True)
            model_result_queue.put(Exception(f"Generation error: {e}"))

    print("[Worker] Model worker thread exiting", flush=True)


# ----- Audio Generation Function -----
async def audio_generation(text: str, websocket: WebSocket):
    global is_speaking, reference_segments

    if not audio_gen_lock.acquire(blocking=False):
        await websocket.send_json(
            {"type": "error", "message": "Audio generation busy, please wait"}
        )
        return

    try:
        is_speaking = True

        await websocket.send_json({"type": "audio_status", "status": "generating"})

        # Preprocess text
        print(f"[Preprocessing] Original text length: {len(text)}", flush=True)
        text = preprocess_text_for_tts(text.lower())
        print(f"[Preprocessing] Cleaned text length: {len(text)}", flush=True)

        # Split long text into chunks
        text_chunks = split_long_text(text, max_words=80)
        print(f"[Preprocessing] Split into {len(text_chunks)} chunks", flush=True)

        if len(text_chunks) > 1:
            await websocket.send_json(
                {
                    "type": "status",
                    "message": f"Long text detected. Splitting into {len(text_chunks)} parts...",
                }
            )

        # Process each chunk
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                await websocket.send_json(
                    {
                        "type": "status",
                        "message": f"Processing part {i+1}/{len(text_chunks)}...",
                    }
                )

            # Estimate audio length for better streaming
            words = chunk.split()
            avg_wpm = 90
            words_per_second = avg_wpm / 60
            padding_seconds = 2
            estimated_seconds = len(words) / words_per_second
            max_audio_length_ms = int((estimated_seconds + padding_seconds) * 1000)

            # Send request to model thread
            model_queue.put(
                (
                    chunk,
                    config.voice_speaker_id,
                    reference_segments,
                    max_audio_length_ms,
                    0.8,  # temperature
                    50,  # topk
                )
            )

            chunk_counter = 0

            # Stream audio chunks in real-time - MAIN CODE
            while True:
                try:
                    result = model_result_queue.get(timeout=1.0)

                    if result == "EOS_AUDIO":
                        break
                        # Add silence after EVERY chunk (except the last one)
                        if i < len(text_chunks) - 1:
                            silence_duration = 0.5  # seconds between chunks
                            silence_samples = int(
                                generator.sample_rate * silence_duration
                            )
                            silence = torch.zeros(silence_samples)
                            silence_array = silence.cpu().numpy().astype(np.float32)

                            await websocket.send_json(
                                {
                                    "type": "audio_chunk",
                                    "audio": silence_array.tolist(),
                                    "sample_rate": generator.sample_rate,
                                    "chunk_num": chunk_counter,
                                    "part": f"{i+1}/{len(text_chunks)}",
                                    "is_silence": True,
                                }
                            )
                            print(
                                f"[Silence] Added {silence_duration}s pause after chunk {i+1}"
                            )
                        break

                    elif isinstance(result, Exception):
                        raise result

                    # Convert to numpy and send audio immediately
                    else:
                        chunk_array = result.cpu().numpy().astype(np.float32)
                        gain = 1.5  # try 1.2–2.0 for louder output
                        chunk_array = np.clip(chunk_array * gain, -1.0, 1.0)

                        await websocket.send_json(
                            {
                                "type": "audio_chunk",
                                "audio": chunk_array.tolist(),
                                "sample_rate": generator.sample_rate,
                                "chunk_num": chunk_counter,
                                "part": (
                                    f"{i+1}/{len(text_chunks)}"
                                    if len(text_chunks) > 1
                                    else None
                                ),
                            }
                        )
                        chunk_counter += 1

                except queue.Empty:
                    continue

        # All chunks processed
        await websocket.send_json(
            {"type": "status", "message": "All parts processed successfully"}
        )

    except Exception as e:
        print(f"[Audio Generation] Error: {e}", flush=True)
        await websocket.send_json(
            {"type": "error", "message": f"Generation failed: {str(e)}"}
        )

    finally:
        is_speaking = False
        audio_gen_lock.release()

        # Send end of stream
        await websocket.send_json({"type": "audio_status", "status": "complete"})


def preprocess_text_for_tts(text):
    text = text.replace("-", " ")
    # This includes: ; : " '  ~ @ # $ % ^ & * ( ) _ - + = [ ] { } \ | / < >
    pattern = r"[^\w\s.,!?\']"  # <-- Look at this pattern
    # Replace matched punctuation with empty string
    cleaned_text = re.sub(pattern, "", text)
    # normalize multiple spaces to single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)
    # ensure there's a space after punctuation for better speech pacing
    cleaned_text = re.sub(r"([.,!?])(\S)", r"\1 \2", cleaned_text)
    cleaned_text = re.sub(r"([.!?])\s", r"\1, ", cleaned_text)

    print(cleaned_text, flush=True)
    return cleaned_text.strip()


# Add this function to your server code
def split_long_text(text, max_words=100):
    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If this would exceed max words and we have content, finish current chunk
        if current_word_count + word_count > max_words and current_chunk:
            # Find the best break point (nearest sentence end)
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = word_count
        else:
            current_chunk.append(sentence)
            current_word_count += word_count

    # Add the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)

        # If sentence is too long, split it by word count
        if word_count > max_words:
            for i in range(0, word_count, max_words):
                chunk = " ".join(words[i : i + max_words])
                chunks.append(chunk)
        else:
            # Normal sentence as its own chunk
            chunks.append(sentence)

    return chunks


# ----- WebSocket Endpoint -----
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if len(active_connections) > 0:
        await websocket.send_json(
            {"type": "error", "message": "Server busy with another connection"}
        )
        await websocket.close()
        return

    await websocket.accept()
    active_connections.append(websocket)

    # print(f"[WebSocket] Client connected. Active connections: {len(active_connections)}", flush=True)
    # print(f"[WebSocket] Model queue size: {model_queue.qsize()}", flush=True)
    # print(f"[WebSocket] Result queue size: {model_result_queue.qsize()}", flush=True)

    print("[WebSocket] Client connected", flush=True)

    # Wait for model to be ready
    if not model_ready.is_set():
        await websocket.send_json(
            {"type": "status", "message": "Models are loading, please wait..."}
        )
        model_ready.wait()

    await websocket.send_json(
        {"type": "status", "message": "Models are ready! You can start streaming."}
    )

    try:
        while True:
            data = await websocket.receive_json()

            if data["type"] == "text_message":
                text = data["text"]
                print(f"[WebSocket] Received text: {text}")

                # Start audio generation
                await audio_generation(text, websocket)

    except Exception as e:
        print(f"[WebSocket] Error: {e}", flush=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

        # CLEAR QUEUES ON DISCONNECT
        print("[Websocket] Clearing queues on disconnect...", flush=True)

        # Clear model result queue
        while not model_result_queue.empty():
            try:
                model_result_queue.get_nowait()
            except queue.Empty:
                break

        # Clear model request queue
        while not model_queue.empty():
            try:
                model_queue.get_nowait()
            except queue.Empty:
                break

        print("[WebSocket] Client disconnected - queues cleared", flush=True)


# ----- Root Endpoint -----
@app.get("/")
def root():
    return {"status": "ready", "model_loaded": model_ready.is_set()}


# ----- Health Check -----
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_ready": model_ready.is_set(),
        "is_speaking": is_speaking,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        timeout_keep_alive=300,  # 5 minutes
        timeout_graceful_shutdown=60,  # 1 minute
    )
