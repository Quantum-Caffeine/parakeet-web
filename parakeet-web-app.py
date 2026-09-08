import platform
import subprocess
import sys
import time
import queue
import threading
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import sherpa_onnx
from ten_vad import TenVad

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse
import uvicorn

# ==========================================
# CONFIGURATION ET CHEMINS (Compatible PyInstaller)
# ==========================================
if getattr(sys, 'frozen', False):
    APP_DIR = Path(sys.executable).resolve().parent
else:
    APP_DIR = Path(__file__).resolve().parent

MODEL_DIR = APP_DIR / "models" / "desktop"
LOGS_DIR = APP_DIR / "transcriptions"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

mic_queue = queue.Queue()
system_queue = queue.Queue()
log_queue = queue.Queue()

app_loop = None
recording_active = False
current_log_file = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global app_loop
    app_loop = asyncio.get_running_loop()
    yield

app = FastAPI(title="Parakeet-Granola Web", lifespan=lifespan)

# ==========================================
# GESTIONNAIRE WEBSOCKET
# ==========================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                self.active_connections.remove(connection)

manager = ConnectionManager()

# ==========================================
# LE VAD (Micro)
# ==========================================
class _SpeechSegment:
    __slots__ = ("samples",)
    def __init__(self, samples: list[float]):
        self.samples = samples

class TenVadDetector:
    def __init__(self, threshold: float = 0.5, min_silence_duration: float = 0.25,
                 min_speech_duration: float = 0.25, max_speech_duration: float = 30.0,
                 sample_rate: int = 16000):
        self._hop_size = 256
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._min_silence_samples = int(min_silence_duration * sample_rate)
        self._min_speech_samples = int(min_speech_duration * sample_rate)
        self._max_speech_samples = int(max_speech_duration * sample_rate)

        self._vad = TenVad(hop_size=self._hop_size, threshold=threshold)
        self._buffer = []
        self._int16_remainder = np.array([], dtype=np.int16)
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._segments = []

    def accept_waveform(self, samples: list[float]):
        arr = np.array(samples, dtype=np.float32)
        int16_data = (arr * 32767).astype(np.int16)
        if len(self._int16_remainder) > 0:
            int16_data = np.concatenate([self._int16_remainder, int16_data])

        i = 0
        while i + self._hop_size <= len(int16_data):
            chunk = int16_data[i:i + self._hop_size]
            prob, _ = self._vad.process(chunk)
            is_speech = prob >= self._threshold
            float_chunk = samples[i:i + self._hop_size] if i + self._hop_size <= len(samples) else arr[i:i + self._hop_size].tolist()

            if is_speech:
                self._silence_samples = 0
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_samples = 0
                self._buffer.extend(float_chunk)
                self._speech_samples += self._hop_size
                if self._speech_samples >= self._max_speech_samples:
                    self._emit_segment()
            else:
                if self._in_speech:
                    self._buffer.extend(float_chunk)
                    self._silence_samples += self._hop_size
                    if self._silence_samples >= self._min_silence_samples:
                        self._emit_segment()
            i += self._hop_size
        self._int16_remainder = int16_data[i:]

    def _emit_segment(self):
        if len(self._buffer) >= self._min_speech_samples:
            self._segments.append(_SpeechSegment(list(self._buffer)))
        self._buffer.clear()
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0

    def empty(self) -> bool:
        return len(self._segments) == 0

    @property
    def front(self) -> _SpeechSegment:
        return self._segments[0]

    def pop(self):
        self._segments.pop(0)

def select_audio_devices():
    devices = sd.query_devices()
    mic_idx = sd.default.device[0]
    sys_idx = 25

    current_os = platform.system()
    if current_os == "Linux":
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if "mix-ioaudio.monitor" in name and dev['max_input_channels'] > 0:
                sys_idx = i
                break
    elif current_os == "Windows":
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            if ("cable output" in name or "loopback" in name or "stéréo mix" in name) and dev['max_input_channels'] > 0:
                sys_idx = i
                break
        if sys_idx is None:
            sys_idx = mic_idx

    return mic_idx, sys_idx

def load_parakeet():
    print("🧠 Chargement du modèle Parakeet (sherpa-onnx)...")
    if not MODEL_DIR.exists():
        print(f"❌ Erreur : Le dossier du modèle est introuvable : {MODEL_DIR}")
        sys.exit(1)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(MODEL_DIR / "encoder.int8.onnx"),
        decoder=str(MODEL_DIR / "decoder.int8.onnx"),
        joiner=str(MODEL_DIR / "joiner.int8.onnx"),
        tokens=str(MODEL_DIR / "tokens.txt"),
        num_threads=2,
        sample_rate=SAMPLE_RATE,
        feature_dim=128,
        model_type="nemo_transducer",
        decoding_method="greedy_search"
    )

def write_log_worker():
    global current_log_file
    while True:
        item = log_queue.get()
        if item is None:
            break
        speaker, text = item
        time_str = datetime.now().strftime("[%H:%M:%S]")
        line = f"{time_str} [{speaker}] {text}"

        if current_log_file and recording_active:
            with open(current_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

        print(line)
        if app_loop and app_loop.is_running():
            asyncio.run_coroutine_threadsafe(manager.broadcast(line), app_loop)

def mic_callback(indata, frames, time_info, status):
    if recording_active:
        mic_queue.put(indata.reshape(-1).tolist())

def system_callback(indata, frames, time_info, status):
    if recording_active:
        system_queue.put(indata.reshape(-1).tolist())

def process_microphone_stream(recognizer):
    vad = TenVadDetector(threshold=0.5, sample_rate=SAMPLE_RATE)
    while True:
        while not mic_queue.empty():
            vad.accept_waveform(mic_queue.get())
        while not vad.empty():
            segment = vad.front
            if recording_active:
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, segment.samples)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    log_queue.put(("Moi", text))
            vad.pop()
        time.sleep(0.02)

def process_system_stream(recognizer):
    buffer = []
    chunk_size = int(SAMPLE_RATE * 5.0)
    while True:
        while not system_queue.empty():
            buffer.extend(system_queue.get())
        if len(buffer) >= chunk_size:
            segment = buffer[:chunk_size]
            buffer = buffer[int(SAMPLE_RATE * 4.0):]

            if recording_active:
                stream = recognizer.create_stream()
                stream.accept_waveform(SAMPLE_RATE, segment)
                recognizer.decode_stream(stream)
                text = stream.result.text.strip()
                if text:
                    log_queue.put(("Réunion", text))
        time.sleep(0.05)

# ==========================================
# API DE CONTRÔLE
# ==========================================
@app.post("/api/start")
async def api_start(title: str = Query("Transcription")):
    global recording_active, current_log_file

    clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
    if not clean_title:
        clean_title = "Transcription"

    date_str = datetime.now().strftime("%Y-%m-%d - %Hh%M")
    current_log_file = LOGS_DIR / f"{clean_title} - {date_str}.txt"

    with open(current_log_file, "w", encoding="utf-8") as f:
        f.write(f"--- Début de la transcription : {title} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

    recording_active = True
    await manager.broadcast("CLEAR_SCREEN")
    await manager.broadcast(f"[{datetime.now().strftime('%H:%M:%S')}] [Système] Enregistrement démarré -> transcriptions/{current_log_file.name}")
    return {"status": "started", "file": str(current_log_file)}

@app.post("/api/pause")
async def api_pause():
    global recording_active
    recording_active = False
    await manager.broadcast(f"[{datetime.now().strftime('%H:%M:%S')}] [Système] Transcription en pause.")
    return {"status": "paused"}

@app.post("/api/resume")
async def api_resume():
    global recording_active
    recording_active = True
    await manager.broadcast(f"[{datetime.now().strftime('%H:%M:%S')}] [Système] Transcription reprise.")
    return {"status": "resumed"}

@app.post("/api/stop")
async def api_stop():
    global recording_active, current_log_file
    recording_active = False
    saved_file = current_log_file
    current_log_file = None
    if saved_file:
        await manager.broadcast(f"[{datetime.now().strftime('%H:%M:%S')}] [Système] Transcription terminée et sauvegardée dans : transcriptions/{Path(saved_file).name}")
    return {"status": "stopped", "file": str(saved_file) if saved_file else None}

@app.post("/api/shutdown")
async def api_shutdown():
    await manager.broadcast(f"[{datetime.now().strftime('%H:%M:%S')}] [Système] Fermeture de l'application...")
    def kill_app():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=kill_app, daemon=True).start()
    return {"status": "shutting down"}

# ==========================================
# INTERFACE WEB HTML / JS (Avec Quitter permanent)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Parakeet - Notes de Réunion</title>
    <style>
        body { font-family: sans-serif; background: #1e1e1e; color: #d4d4d4; padding: 20px; max-width: 800px; margin: auto; }

        .header-bar { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px; }
        h2 { color: #4ec9b0; margin: 0; }

        .panel { background: #252526; padding: 15px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #333; }
        .config-group { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; }
        .config-group label { font-weight: bold; color: #9cdcfe; }
        .config-group input { flex: 1; padding: 9px; background: #3c3c3c; border: 1px solid #555; color: white; border-radius: 4px; font-size: 14px; }

        .controls { display: flex; gap: 10px; align-items: center; }
        button { padding: 10px 18px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px; transition: opacity 0.2s; }
        button:hover { opacity: 0.85; }

        .btn-record { background: #d16969; color: white; width: 100%; font-size: 16px; padding: 12px; }
        .btn-pause { background: #dcdcaa; color: #1e1e1e; }
        .btn-resume { background: #4ec9b0; color: #1e1e1e; }
        .btn-stop { background: #555; color: white; }
        .btn-quit { background: #512b2b; color: #ff9999; border: 1px solid #d16969; font-size: 13px; padding: 6px 12px; }

        #log { background: #2d2d2d; border: 1px solid #444; padding: 15px; height: 400px; overflow-y: scroll; font-family: monospace; border-radius: 5px; }
        .moi { color: #569cd6; margin-bottom: 6px; }
        .reunion { color: #ce9178; margin-bottom: 6px; }
        .system { color: #dcdcaa; font-style: italic; margin-bottom: 6px; }
    </style>
</head>
<body>
    <div class="header-bar">
        <h2>🎙️ Parakeet - Live Notes</h2>
        <button class="btn-quit" onclick="quitApp()">🚪 Quitter l'application</button>
    </div>

    <div class="panel">
        <!-- État Initial : Champ de nom vierge + Bouton Rouge -->
        <div id="initial-view">
            <div class="config-group">
                <label for="logTitle">Nom :</label>
                <input type="text" id="logTitle" placeholder="Ex: Réunion de Gestion">
            </div>
            <button class="btn-record" onclick="startTranscription()">🔴 Débuter Transcription</button>
        </div>

        <!-- État Actif : Boutons Pause/Reprendre & Fin -->
        <div id="active-view" style="display: none;" class="controls">
            <button id="pause-btn" class="btn-pause" onclick="togglePause()">⏸️ Pause</button>
            <button class="btn-stop" onclick="stopTranscription()">⏹️ Fin de transcription</button>
        </div>
    </div>

    <div id="log">En attente du lancement d'une transcription...</div>

    <script>
        const ws = new WebSocket("ws://" + window.location.host + "/ws");
        const logDiv = document.getElementById("log");
        let isPaused = false;

        function startTranscription() {
            const titleInput = document.getElementById("logTitle").value;
            const title = titleInput.trim() !== "" ? titleInput : "Transcription";

            fetch(`/api/start?title=${encodeURIComponent(title)}`, {method: 'POST'})
                .then(res => res.json())
                .then(data => {
                    if(data.status === "started") {
                        document.getElementById("initial-view").style.display = "none";
                        document.getElementById("active-view").style.display = "flex";
                        isPaused = false;
                        updatePauseButton();
                    }
                });
        }

        function togglePause() {
            if (!isPaused) {
                fetch('/api/pause', {method: 'POST'}).then(() => {
                    isPaused = true;
                    updatePauseButton();
                });
            } else {
                fetch('/api/resume', {method: 'POST'}).then(() => {
                    isPaused = false;
                    updatePauseButton();
                });
            }
        }

        function updatePauseButton() {
            const btn = document.getElementById("pause-btn");
            if (isPaused) {
                btn.textContent = "▶️ Reprendre Transcription";
                btn.className = "btn-resume";
            } else {
                btn.textContent = "⏸️ Pause";
                btn.className = "btn-pause";
            }
        }

        function stopTranscription() {
            fetch('/api/stop', {method: 'POST'})
                .then(res => res.json())
                .then(data => {
                    document.getElementById("active-view").style.display = "none";
                    document.getElementById("initial-view").style.display = "block";
                    document.getElementById("logTitle").value = "";
                });
        }

        function quitApp() {
            if (confirm('Voulez-vous vraiment fermer l\\'application et couper le serveur ?')) {
                fetch('/api/shutdown', {method: 'POST'}).then(() => {
                    document.body.innerHTML = "<h2 style='text-align:center; margin-top:30vh; color:#4ec9b0;'>🛑 Application fermée.<br><span style='font-size:16px; color:#aaa; font-weight:normal;'>Vous pouvez fermer cet onglet en toute sécurité.</span></h2>";
                });
            }
        }

        ws.onmessage = function(event) {
            const line = event.data;
            if (line === "CLEAR_SCREEN") {
                logDiv.innerHTML = "";
                return;
            }
            if (logDiv.innerHTML.includes("En attente")) {
                logDiv.innerHTML = "";
            }
            const p = document.createElement("div");

            if (line.includes("[Moi]")) {
                p.className = "moi";
            } else if (line.includes("[Réunion]")) {
                p.className = "reunion";
            } else if (line.includes("[Système]")) {
                p.className = "system";
            }
            p.textContent = line;
            logDiv.appendChild(p);
            logDiv.scrollTop = logDiv.scrollHeight;
        };
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get():
    return HTML_PAGE

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ==========================================
# POINT D'ENTRÉE & LANCEMENT
# ==========================================
if __name__ == "__main__":
    current_os = platform.system()

    recognizer = load_parakeet()
    mic_device, sys_device = select_audio_devices()
    print(f"🎙️ Micro ID : {mic_device} | Système ID : {sys_device}")

    threading.Thread(target=write_log_worker, daemon=True).start()

    chunk_duration = 0.1
    samples_per_chunk = int(SAMPLE_RATE * chunk_duration)

    def start_audio_engine():
        with sd.InputStream(device=mic_device, channels=1, samplerate=SAMPLE_RATE, blocksize=samples_per_chunk, callback=mic_callback), \
             sd.InputStream(device=sys_device, channels=1, samplerate=SAMPLE_RATE, blocksize=samples_per_chunk, callback=system_callback):

            threading.Thread(target=process_microphone_stream, args=(recognizer,), daemon=True).start()
            threading.Thread(target=process_system_stream, args=(recognizer,), daemon=True).start()

            while True:
                time.sleep(1)

    threading.Thread(target=start_audio_engine, daemon=True).start()

    print(f"\n📁 Dossier des transcriptions : {LOGS_DIR}")
    print("🌐 Serveur web démarré sur : http://localhost:8000")

    import webbrowser
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Arrêt propre du serveur et des flux audio.")
