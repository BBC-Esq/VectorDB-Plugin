import gc
import tempfile
from pathlib import Path

import psutil
import sounddevice as sd
import numpy as np
import soundfile as sf
import torch
from PySide6.QtCore import QThread, Signal

import whisper_s2t
from utilities import my_cprint

def get_logical_core_count():
    return psutil.cpu_count(logical=False)

CPU_THREADS = max(4, get_logical_core_count() - 8)
DEVICE = "cpu"
COMPUTE_TYPE = "float32"
MODEL_IDENTIFIER = "ctranslate2-4you/distil-whisper-small.en-ct2-float32"

class TranscriptionThread(QThread):
    transcription_complete = Signal(str)

    def __init__(self, audio_file, voice_recorder):
        super().__init__()
        self.audio_file = audio_file
        self.voice_recorder = voice_recorder

    def run(self):            
        

        model_kwargs = {
            'compute_type': COMPUTE_TYPE,
            'model_identifier': MODEL_IDENTIFIER,
            'backend': 'CTranslate2',
            "device": DEVICE,
            "cpu_threads": CPU_THREADS,
        }
        self.model = whisper_s2t.load_model(**model_kwargs)
        my_cprint("Whisper model loaded.", 'green')
        
        out = self.model.transcribe_with_vad([self.audio_file],
                                             lang_codes=['en'],
                                             tasks=['transcribe'],
                                             initial_prompts=[None],
                                             batch_size=4)
        
        transcription_text = " ".join(item['text'] for item in out[0]).strip()
        
        self.transcription_complete.emit(transcription_text)
        
        Path(self.audio_file).unlink()
        # self.voice_recorder.ReleaseTranscriber()

        del self.model
        # my_cprint("Whisper model removed from memory.", 'red')

class RecordingThread(QThread):
    def __init__(self, voice_recorder):
        super().__init__()
        self.voice_recorder = voice_recorder

    def run(self):
        self.voice_recorder.record_audio()

class VoiceRecorder:
    def __init__(self, gui_instance, channels=1, rate=16000, chunk=1024):
        self.gui_instance = gui_instance
        self.channels, self.rate, self.chunk = channels, rate, chunk
        self.is_recording, self.frames = False, []
        self.recording_thread = None
        self.transcription_thread = None

    def record_audio(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.frames.append(indata.copy())

        try:
            with sd.InputStream(samplerate=self.rate, channels=self.channels, 
                              callback=callback, blocksize=self.chunk):
                while self.is_recording:
                    sd.sleep(100)
        except sd.PortAudioError as e:
            my_cprint(f"Audio recording error: {str(e)}", 'red')
            self.is_recording = False
            self.gui_instance.update_transcription("Error: Failed to access microphone")

    def save_audio(self):
        self.is_recording = False
        if not self.frames:
            my_cprint("No audio data recorded.", 'yellow')
            return

        temp_file = Path(tempfile.mktemp(suffix=".wav"))
        audio_data = np.concatenate(self.frames, axis=0)
        sf.write(str(temp_file), audio_data, self.rate)
        self.frames.clear()

        if temp_file.stat().st_size < 1024:
            my_cprint("Recording too short, discarding.", 'yellow')
            temp_file.unlink()
            return

        self.transcription_thread = TranscriptionThread(str(temp_file), self)
        self.transcription_thread.transcription_complete.connect(self.gui_instance.update_transcription)
        self.transcription_thread.start()

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_thread = RecordingThread(self)
            self.recording_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.recording_thread is not None:
            self.recording_thread.wait()
            self.save_audio()

    def ReleaseTranscriber(self):
        if hasattr(self, 'model'):
            if hasattr(self.model, 'model'):
                del self.model.model
            if hasattr(self.model, 'feature_extractor'):
                del self.model.feature_extractor
            if hasattr(self.model, 'hf_tokenizer'):
                del self.model.hf_tokenizer
            del self.model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()