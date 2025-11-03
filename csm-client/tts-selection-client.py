# needs tts-selection-server.py running and also config_selection.json!
# multi voice code based on ID's to use multiple voices at once

import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional

# ===== GLOBAL CONFIGURATION =====
SERVER_URL = "ws://selection.hussainkazarani.site/ws"
# ================================


class TTSClient:
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.sample_rate = 24000
        self.available_voices = {}
        self.default_voice_id = 0
        self.playback_thread = None
        self.stop_playback = threading.Event()

    async def connect(self):
        """Connect to the TTS WebSocket server"""
        try:
            print(f"Connecting to {self.server_url}...")
            self.websocket = await websockets.connect(
                self.server_url, ping_interval=30, ping_timeout=10, close_timeout=10
            )
            print("✓ Connected to server")

            # Wait for initial messages
            await self._receive_initial_messages()

        except Exception as e:
            print(f"✗ Connection failed: {e}")
            raise

    async def _receive_initial_messages(self):
        """Receive and process initial server messages"""
        try:
            while True:
                message = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
                data = json.loads(message)

                if data["type"] == "available_voices":
                    self.available_voices = data["voices"]
                    self.default_voice_id = data["default_voice"]
                    print(f"✓ Available voices: {self.available_voices}")
                    print(f"  Default voice ID: {self.default_voice_id}")

                elif data["type"] == "status":
                    print(f"  {data['message']}")
                    if "ready" in data["message"].lower():
                        break

        except asyncio.TimeoutError:
            print("✗ Timeout waiting for server ready message")
            raise

    def _audio_playback_worker(self):
        """Worker thread for audio playback"""
        print("[Playback] Worker thread started")

        while not self.stop_playback.is_set():
            try:
                audio_data = self.audio_queue.get(timeout=0.1)

                if audio_data is None:  # Stop signal
                    break

                # Play audio
                sd.play(audio_data, self.sample_rate)
                sd.wait()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Playback] Error: {e}")

        print("[Playback] Worker thread stopped")

    def start_playback_thread(self):
        """Start the audio playback thread"""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.stop_playback.clear()
            self.playback_thread = threading.Thread(
                target=self._audio_playback_worker, daemon=True, name="audio_playback"
            )
            self.playback_thread.start()

    def stop_playback_thread(self):
        """Stop the audio playback thread"""
        self.stop_playback.set()
        self.audio_queue.put(None)  # Signal to stop
        if self.playback_thread:
            self.playback_thread.join(timeout=2.0)

    async def generate_speech(self, text: str, voice_id: Optional[int] = None):
        """Generate speech from text"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        if voice_id is None:
            voice_id = self.default_voice_id

        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Start playback thread
        self.start_playback_thread()

        # Send text to server
        message = {"type": "text_message", "text": text, "voice_id": voice_id}

        print(f"\n→ Sending text (voice {voice_id}): {text[:100]}...")
        await self.websocket.send(json.dumps(message))

        # Receive and process responses
        chunk_count = 0
        self.is_playing = True

        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data["type"] == "audio_chunk":
                    chunk_count += 1
                    audio_array = np.array(data["audio"], dtype=np.float32)
                    self.sample_rate = data["sample_rate"]

                    # Queue audio for playback
                    self.audio_queue.put(audio_array)

                    part_info = f" (part {data['part']})" if data.get("part") else ""
                    print(
                        f"  ← Received chunk {data['chunk_num']}{part_info}", end="\r"
                    )

                elif data["type"] == "audio_status":
                    if data["status"] == "generating":
                        print("  Status: Generating audio...")
                    elif data["status"] == "first_chunk":
                        print("  Status: Playback started")

                elif data["type"] == "status":
                    print(f"\n  {data['message']}")

                elif data["type"] == "completion":
                    print(f"\n✓ {data['message']} ({data['chunks_processed']} chunks)")
                    break

                elif data["type"] == "error":
                    print(f"\n✗ Error: {data['message']}")
                    break

        finally:
            self.is_playing = False

            # Wait for audio queue to empty
            while not self.audio_queue.empty():
                await asyncio.sleep(0.1)

            print("  Audio playback complete\n")

    async def close(self):
        """Close the connection"""
        self.stop_playback_thread()

        if self.websocket:
            await self.websocket.close()
            print("✓ Connection closed")

    def list_voices(self):
        """Print available voices"""
        print("\nAvailable voices:")
        for voice_id, voice_name in self.available_voices.items():
            default_marker = (
                " (default)" if int(voice_id) == self.default_voice_id else ""
            )
            print(f"  {voice_id}: {voice_name}{default_marker}")
        print()


async def interactive_mode(client: TTSClient):
    """Interactive mode for testing"""
    print("\n" + "=" * 60)
    print("TTS Client - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /voices - List available voices")
    print("  /voice <id> - Set voice ID for next generation")
    print("  /quit - Exit")
    print("  <text> - Generate speech from text")
    print("=" * 60 + "\n")

    current_voice = client.default_voice_id

    while True:
        try:
            user_input = input(f"[Voice {current_voice}] > ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("Exiting...")
                break

            elif user_input == "/voices":
                client.list_voices()

            elif user_input.startswith("/voice "):
                try:
                    voice_id = int(user_input.split()[1])
                    if str(voice_id) in client.available_voices:
                        current_voice = voice_id
                        print(
                            f"✓ Voice set to {voice_id}: {client.available_voices[str(voice_id)]}"
                        )
                    else:
                        print(
                            f"✗ Invalid voice ID. Available: {list(client.available_voices.keys())}"
                        )
                except (ValueError, IndexError):
                    print("✗ Usage: /voice <id>")

            else:
                await client.generate_speech(user_input, current_voice)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"✗ Error: {e}")


async def main():
    # Create client (uses global SERVER_URL)
    client = TTSClient()

    try:
        # Connect to server
        await client.connect()

        # Run interactive mode
        await interactive_mode(client)

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        await client.close()


if __name__ == "__main__":
    # Required for sounddevice on some systems
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()
