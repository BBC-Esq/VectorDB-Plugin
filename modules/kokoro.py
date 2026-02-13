import sys
import os
from pathlib import Path
import queue
import threading
import re
import torch
import sounddevice as sd
import warnings
import logging
from typing import Optional, Union

class KokoroTTS:
    VOICES = [
        'af',
        'af_bella',
        'af_sarah',
        'am_adam',
        'am_michael',
        'bf_emma',
        'bf_isabella',
        'bm_george',
        'bm_lewis',
        'af_nicole',
        'af_sky'
    ]
    
    def __init__(self, repo_path: str):
        self.REPO_PATH = Path(repo_path)
        sys.path.append(str(self.REPO_PATH))

        from models import build_model
        from kokoro import generate, generate_full, phonemize
        self.generate = generate
        self.generate_full = generate_full
        self.phonemize = phonemize

        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.model = None
        self.voicepack_cache = {}
        self.current_voice_name = None

        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

    def _load_model_and_voice(self, voice_name: str):
        device = 'cpu'
        
        if self.model is None:
            model_path = self.REPO_PATH / 'kokoro-v0_19.pth'
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            from models import build_model
            self.model = build_model(str(model_path), device)

        if voice_name not in self.voicepack_cache:
            voices_path = self.REPO_PATH / 'voices'
            if not voices_path.exists():
                raise FileNotFoundError(f"Voices directory not found at {voices_path}")
            
            voicepack_path = voices_path / f'{voice_name}.pt'
            if not voicepack_path.exists():
                raise FileNotFoundError(f"Voice file not found at {voicepack_path}")
                
            self.voicepack_cache[voice_name] = torch.load(str(voicepack_path), weights_only=True).to(device)
            print(f"Loaded voicepack for {voice_name}")

        self.current_voice_name = voice_name

    @staticmethod
    def _drain_queue(q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def stop(self):
        self.stop_event.set()
        self._drain_queue(self.sentence_queue)
        self._drain_queue(self.audio_queue)
        

    def _process_sentences(self, speed: float, force_accent: Optional[str]):
        while not self.stop_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=1)
                if sentence is None:
                    self.audio_queue.put(None)
                    break

                if self.stop_event.is_set():
                    break

                lang = force_accent if force_accent else self.current_voice_name[0]
                
                logging.debug("About to generate phonemes...")
                ps = self.phonemize(sentence, lang)
                logging.debug(f"Generated phonemes: {ps}")

                if self.stop_event.is_set():
                    break

                try:
                    voicepack = self.voicepack_cache[self.current_voice_name]
                    
                    audio, phonemes = self.generate_full(
                        self.model, 
                        sentence, 
                        voicepack, 
                        lang=lang, 
                        speed=speed,
                        ps=ps
                    )

                    if audio is not None and not self.stop_event.is_set():
                        self.audio_queue.put(audio)
                    elif audio is None:
                        print(f"Failed to generate audio for sentence: {sentence}")

                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Error generating speech for sentence: {str(e)}")
                        print(f"Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                    continue

            except queue.Empty:
                continue

    def _play_audio(self):
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=1)
                if audio is None:
                    break
                
                if self.stop_event.is_set():
                    break
                    
                try:
                    sd.play(audio, 24000)
                    
                    while sd.get_stream().active and not self.stop_event.is_set():
                        sd.sleep(50)
                    
                    if self.stop_event.is_set():
                        try:
                            sd.stop()
                        except Exception as e:
                            print(f"Safe stop error (expected): {e}")
                        break
                    else:
                        sd.wait()
                        
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Audio playback error: {e}")
                    try:
                        sd.stop()
                    except:
                        pass
                    break
                
            except queue.Empty:
                continue
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Audio queue error: {e}")
                break

    def speak(self, 
             text: str, 
             voice: str = 'bm_george',
             speed: float = 1.3,
             force_accent: Optional[str] = None) -> None:
        if voice not in self.VOICES:
            raise ValueError(f"Invalid voice. Choose from: {self.VOICES}")

        if force_accent and force_accent not in ['a', 'b']:
            raise ValueError("force_accent must be 'a' for American, 'b' for British, or None")

        if speed <= 0:
            raise ValueError("Speed must be positive")

        self.stop_event.clear()
        self.sentence_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        self._load_model_and_voice(voice)

        sentences = [s.strip() for s in re.split(r'[.!?;]+\s*', text) if s.strip()]

        process_thread = threading.Thread(
            target=self._process_sentences,
            args=(speed, force_accent)
        )
        playback_thread = threading.Thread(target=self._play_audio)

        process_thread.daemon = True
        playback_thread.daemon = True

        process_thread.start()
        playback_thread.start()

        for sentence in sentences:
            if self.stop_event.is_set():
                break
            self.sentence_queue.put(sentence)
        
        if not self.stop_event.is_set():
            self.sentence_queue.put(None)

        process_thread.join()
        playback_thread.join()
