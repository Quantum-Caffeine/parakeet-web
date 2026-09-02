import platform
import subprocess
import sys
import time
import queue
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
import sherpa_onnx
from ten_vad import TenVad

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_RATE = 16000
LOG_FILE = f"notes_reunion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models" / "desktop"

mic_queue = queue.Queue()
system_queue = queue.Queue()
log_queue = queue.Queue()

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

# ==========================================
# SÉLECTION UNIVERSELLE DES PÉRIPHÉRIQUES
# ==========================================
def select_audio_devices():
    devices = sd.query_devices()

    mic_idx = None
    sys_idx = None

    # 1. Recherche du micro (par défaut ou via un micro USB disponible)
    mic_idx = sd.default.device[0]

    # 2. Recherche du canal système (notre mixeur virtuel universel "Mix-IOAudio.monitor")
    for i, dev in enumerate(devices):
        name = dev['name'].lower()
        if "mix-ioaudio.monitor" in name and dev['max_input_channels'] > 0:
            sys_idx = i
            break

    # Fallback de sécurité si le mixeur virtuel n'est pas encore vu par sounddevice
    if sys_idx is None:
        for i, dev in enumerate(devices):
            if "monitor" in name and dev['max_input_channels'] > 0:
                sys_idx = i
                break

    if sys_idx is None:
        sys_idx = mic_idx # Dernier recours

    return mic_idx, sys_idx

# ==========================================
# CHARGEMENT PARAKEET
# ==========================================
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
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"--- Début de la session : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n\n")

    while True:
        speaker, text = log_queue.get()
        if speaker is None:
            break
        time_str = datetime.now().strftime("[%H:%M:%S]")
        line = f"{time_str} [{speaker}] {text}\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        print(line.strip())

def mic_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    mic_queue.put(indata.reshape(-1).tolist())

def system_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    system_queue.put(indata.reshape(-1).tolist())

import threading

def process_microphone_stream(recognizer):
    vad = TenVadDetector(threshold=0.5, sample_rate=SAMPLE_RATE)
    while True:
        while not mic_queue.empty():
            vad.accept_waveform(mic_queue.get())
        while not vad.empty():
            segment = vad.front
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

            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment)
            recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                log_queue.put(("Réunion", text))
        time.sleep(0.05)

if __name__ == "__main__":
    print(f"🚀 Démarrage du moteur double-canal...")
    recognizer = load_parakeet()

    # On recrée notre pont virtuel PipeWire pour être sûr que le son système est routé
    subprocess.run(["pactl", "load-module", "module-null-sink", "sink_name=Mix-IOAudio", "sink_properties=device.description=Mix-IOAudio"], capture_output=True)
    subprocess.run(["pactl", "load-module", "module-loopback", "source=@DEFAULT_SINK@.monitor", "sink=Mix-IOAudio"], capture_output=True)
    time.sleep(1)

    mic_device, sys_device = select_audio_devices()
    print(f"🎙️ Sélection finale -> Micro ID : {mic_device} | Système/Réunion ID : {sys_device}")

    log_thread = threading.Thread(target=write_log_worker, daemon=True)
    log_thread.start()

    chunk_duration = 0.1
    samples_per_chunk = int(SAMPLE_RATE * chunk_duration)

    try:
        with sd.InputStream(device=mic_device, channels=1, samplerate=SAMPLE_RATE, blocksize=samples_per_chunk, callback=mic_callback), \
             sd.InputStream(device=sys_device, channels=1, samplerate=SAMPLE_RATE, blocksize=samples_per_chunk, callback=system_callback):

            t_mic = threading.Thread(target=process_microphone_stream, args=(recognizer,), daemon=True)
            t_sys = threading.Thread(target=process_system_stream, args=(recognizer,), daemon=True)
            t_mic.start()
            t_sys.start()

            print("(Écoute active. Lance une vidéo YouTube pour tester le canal Réunion. Ctrl+C pour quitter)")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Arrêt du moteur.")
        log_queue.put((None, None))
