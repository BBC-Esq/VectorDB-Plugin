import queue
import re
import threading
from pathlib import Path

import io
import numpy as np
import sounddevice as sd
import torch
import yaml
from tqdm import tqdm
import ChatTTS
from transformers import AutoProcessor, BarkModel
from whisperspeech.pipeline import Pipeline
import soundfile as sf
from gtts import gTTS
from gtts.tokenizer import pre_processors, tokenizer_cases

from core.utilities import my_cprint
from core.constants import WHISPER_SPEECH_MODELS, PROJECT_ROOT

current_directory = PROJECT_ROOT
CACHE_DIR = current_directory / "models" / "tts"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class BaseAudio:
    def __init__(self):
        self.sentence_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.config = {}
        self.processing_thread = None

    def load_config(self, config_file, section):
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if section in config_data:
                self.config = config_data[section]
            else:
                print(f"Warning: Section '{section}' not found in config file.")
                self.config = {}

    def initialize_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise RuntimeError("CUDA is not available, but it's required for this program.")

    def play_audio_from_queue(self):
        while not self.stop_event.is_set():
            try:
                queue_item = self.audio_queue.get(timeout=5)
                if queue_item is None or self.stop_event.is_set():
                    break
                audio_array, sampling_rate = queue_item
                try:
                    if len(audio_array.shape) == 1:
                        audio_array = np.expand_dims(audio_array, axis=1)
                    elif len(audio_array.shape) == 2 and audio_array.shape[1] != 1:
                        audio_array = audio_array.T
                    sd.play(audio_array, samplerate=sampling_rate)
                    sd.wait()
                except Exception as e:
                    print(f"Error playing audio: {e}")
            except queue.Empty:
                if self.processing_thread is None or not self.processing_thread.is_alive():
                    break

    def run(self, input_text_file):
        try:
            with open(input_text_file, 'r', encoding='utf-8') as file:
                input_text = file.read()
                sentences = re.split(r'[.!?;]+\s*', input_text)
        except Exception as e:
            print(f"Error reading {input_text_file}: {e}")
            return

        self.processing_thread = threading.Thread(target=self.process_text_to_audio, args=(sentences,))
        playback_thread = threading.Thread(target=self.play_audio_from_queue)

        self.processing_thread.daemon = True
        playback_thread.daemon = True

        self.processing_thread.start()
        playback_thread.start()

        self.processing_thread.join()
        playback_thread.join()

    def stop(self):
        self.stop_event.set()
        self.audio_queue.put(None)


class ChatterboxAudio(BaseAudio):

    PITCH_FACTOR = 0.93
    SPEED_FACTOR = 0.93


    def __init__(self):
        super().__init__()

        import torch, warnings
        device_pref = getattr(self, "DEVICE", "auto")
        self.device = self._select_device(device_pref)
        if self.device != "cpu":
            _orig_load = torch.load
            torch.load = lambda *a, **k: _orig_load(
                *a, **{**k, "map_location": k.get("map_location", self.device)}
            )
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        from chatterbox.tts import ChatterboxTTS
        print(f"Loading Chatterbox TTS on [{self.device}] …")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sr = self.model.sr
        print("Model ready!")

        accent = getattr(self, "ACCENT_PRESET", None)
        if accent and hasattr(self, "ACCENT_SETTINGS"):
            style = self.ACCENT_SETTINGS.get(accent, {})
            self.exaggeration = style.get("exaggeration", 0.5)
            self.cfg_weight   = style.get("cfg_weight",   0.5)
        else:
            self.exaggeration = getattr(self, "EXAGGERATION", 0.5)
            self.cfg_weight   = getattr(self, "CFG_WEIGHT",   0.5)

        self.pitch_factor = getattr(self, "PITCH_FACTOR", 1.0)
        self.speed_factor = getattr(self, "SPEED_FACTOR", 1.0)
        self.tone         = getattr(self, "TONE", "neutral")
        self.normalise    = getattr(self, "NORMALISE", True)
        self.int16_output = getattr(self, "INT16_OUTPUT", False)

    def _select_device(self, pref):
        import torch, warnings
        pref = pref.lower()
        if pref == "cpu":
            return "cpu"
        if pref in ("gpu", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
            if getattr(self, "GPU_STRICT", False):
                raise RuntimeError("CUDA requested but unavailable.")
            warnings.warn("CUDA not available – falling back to CPU.", RuntimeWarning)
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _apply_voice_modifications(wav, sr, pitch_factor=1.0, speed_factor=1.0, tone="neutral"):
        try:
            import torch, torchaudio.functional as F
            if pitch_factor != 1.0:
                tgt_sr = int(sr * pitch_factor)
                wav = F.resample(wav, sr, tgt_sr)
                wav = F.resample(wav, tgt_sr, sr)
            if speed_factor != 1.0 and wav.numel():
                tgt_len = int(wav.shape[-1] / speed_factor)
                if tgt_len > 0:
                    wav = torch.nn.functional.interpolate(
                        wav.unsqueeze(0), size=tgt_len, mode="linear", align_corners=False
                    ).squeeze(0)
            if tone == "happy":
                wav *= 1.1
            elif tone == "serious":
                wav *= 0.9
            elif tone == "calm":
                wav = torch.tanh(wav * 0.8)
            elif tone == "excited":
                wav *= 1.2
            return torch.clamp(wav, -1.0, 1.0)
        except Exception as e:
            print(f"Voice-mod skipped: {e}")
            return wav

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        import numpy as np, torch
        for sentence in sentences:
            if not sentence.strip() or self.stop_event.is_set():
                continue
            try:
                wav = self.model.generate(
                    sentence,
                    exaggeration=self.exaggeration,
                    cfg_weight=self.cfg_weight,
                )
            except Exception as e:
                print(f"Generation failed: {e}")
                self.audio_queue.put(None)
                break

            wav = self._apply_voice_modifications(
                wav, self.sr,
                pitch_factor=self.pitch_factor,
                speed_factor=self.speed_factor,
                tone=self.tone,
            )

            audio = wav.squeeze().cpu().numpy().astype(np.float32)
            if self.normalise and audio.size:
                peak = np.max(np.abs(audio))
                if peak:
                    audio /= peak
            if self.int16_output:
                audio = (audio * 32767).astype(np.int16)

            self.audio_queue.put((audio, self.sr))

            if torch.cuda.is_available():
                del wav
                torch.cuda.empty_cache()

        self.audio_queue.put(None)


class BarkAudio(BaseAudio):
    def __init__(self):
        super().__init__()
        self.load_config('config.yaml', 'bark')
        self.initialize_device()
        self.initialize_model_and_processor()

    def initialize_model_and_processor(self):
        repository_id = "suno/bark" if self.config['size'] == 'normal' else f"suno/bark-{self.config['size']}"
        
        self.processor = AutoProcessor.from_pretrained(repository_id, token=False, cache_dir=CACHE_DIR)
        
        self.model = BarkModel.from_pretrained(
            repository_id,
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR,
            token=False
        ).to(self.device)

        self.model.eval()
        
        my_cprint("Bark model loaded (float16)", "green")

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        for sentence in tqdm(sentences, desc="Processing Sentences"):
            if sentence.strip():
                print(f"Processing sentence: {sentence}")
                try:
                    inputs = self.processor(text=sentence, voice_preset=self.config['speaker'], return_tensors="pt")
                    inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                            for k, v in inputs.items()}

                    speech_output = self.model.generate(
                        **inputs,
                        use_cache=True,
                        do_sample=True,
                        pad_token_id=0,
                    )

                    audio_array = speech_output[0].cpu().numpy()
                    audio_array = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
                    self.audio_queue.put((audio_array, self.model.generation_config.sample_rate))
                except Exception as e:
                    print(f"Exception during audio generation: {str(e)}")
                    continue
        self.audio_queue.put(None)


class WhisperSpeechAudio(BaseAudio):
    def __init__(self):
        super().__init__()
        self.load_config('config.yaml', 'tts')
        self.pipe = None
        self.initialize_model()

    def get_whisper_speech_models(self):
        s2a_model = self.config.get('s2a', 's2a-q4-hq-fast-en+pl.model')
        s2a = f"collabora/whisperspeech:{s2a_model}"
        
        t2s_model = self.config.get('t2s', 't2s-base-en+pl.model')
        t2s = f"collabora/whisperspeech:{t2s_model}"

        return s2a, t2s

    def initialize_model(self):
        s2a, t2s = self.get_whisper_speech_models()

        try:
            self.pipe = Pipeline(
                s2a_ref=s2a,
                t2s_ref=t2s,
                cache_dir=CACHE_DIR
            )
            my_cprint(f"{s2a.split(':')[-1]} loaded\n{t2s.split(':')[-1]} loaded.", "green")
        except Exception as e:
            my_cprint(f"Error initializing WhisperSpeech models: {str(e)}", "red")
            self.pipe = None

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        for sentence in tqdm(sentences, desc="Processing Sentences"):
            if sentence and not self.stop_event.is_set():
                try:
                    audio_tensor = self.pipe.generate(sentence)
                    audio_np = (audio_tensor.cpu().numpy() * 32767).astype(np.int16)
                    if len(audio_np.shape) == 1:
                        audio_np = np.expand_dims(audio_np, axis=1)
                    else:
                        audio_np = audio_np.T
                    self.audio_queue.put((audio_np, 24000))
                except Exception as e:
                    my_cprint(f"Error processing sentence: {str(e)}", "red")
        self.audio_queue.put(None)

    def run(self, input_text_file):
        self.initialize_device()
        super().run(input_text_file)


class ChatTTSAudio(BaseAudio):
    def __init__(self):
        super().__init__()
        
        print("Initializing ChatTTSAudio...")
        
        self.initialize_device()
        self.chat = ChatTTS.Chat()

        chattts_dir = CACHE_DIR / "2Noise--ChatTTS"
        chattts_dir.mkdir(parents=True, exist_ok=True)

        self.chat.load(
            source="huggingface",
            device=self.device,
            compile=False,
            use_flash_attn=False,
            local_dir=str(chattts_dir)
        )

        torch.manual_seed(11)
        self.rand_spk = self.chat.sample_random_speaker()

        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.rand_spk,
            temperature=0.7,
            top_P=1,
            top_K=40,
            prompt='[speed_5]'
        )

        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_0][laugh_0][break_0]',
            temperature=0.7,
            top_P=0.7,
            top_K=20
        )

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        print(f"Starting text processing... ({len(sentences)} sentences)")
        for sentence in sentences:
            if not sentence or not sentence.strip():
                continue

            print(f"Processing sentence: {sentence}")
            try:
                wavs = self.chat.infer(
                    sentence,
                    params_refine_text=self.params_refine_text,
                    params_infer_code=self.params_infer_code,
                    split_text=False
                )

                if wavs is not None and len(wavs) > 0:
                    audio_data = wavs[0]
                    if isinstance(audio_data, torch.Tensor):
                        audio_data = audio_data.cpu().numpy()
                    audio_data = audio_data.squeeze()

                    if np.prod(audio_data.shape) > 0:
                        print(f"Audio data shape: {audio_data.shape}")
                        if np.abs(audio_data).max() > 1.0:
                            audio_data = audio_data / np.abs(audio_data).max()
                        print(f"Audio range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")
                        self.audio_queue.put((audio_data, 24000))
                        print("Audio data queued")
            except Exception as e:
                print(f"Error processing sentence: {str(e)}\n{type(e)}")
                import traceback
                print(traceback.format_exc())
                continue

        print("Text processing complete, sending end signal")
        self.audio_queue.put(None)


class GoogleTTSAudio:
    
    def __init__(self, lang='en', slow=False, tld='com', silence_threshold=0.01, max_silence_ms=100):
        self.lang = lang
        self.slow = slow
        self.tld = tld
        self.silence_threshold = silence_threshold
        self.max_silence_ms = max_silence_ms

    def run(self, input_text_file):
        try:
            with open(input_text_file, 'r', encoding='utf-8') as file:
                text = file.read()
        except FileNotFoundError:
            print(f"Error: File not found at {input_text_file}")
            return
        except IOError:
            print(f"Error: Unable to read file at {input_text_file}")
            return

        processed_text = self.preprocess_text(text)
        tokens = self.tokenize_and_minimize(processed_text)

        all_audio_data = []
        samplerate = None
        for token in tokens:
            if token.strip():
                print(f"Processing token: '{token}'")
                fp = io.BytesIO()

                if token.startswith("<continue>"):
                    token = token[10:].strip()

                tts = gTTS(text=token, lang=self.lang, slow=self.slow, tld=self.tld)
                tts.write_to_fp(fp)
                fp.seek(0)
                data, samplerate = sf.read(fp)
                all_audio_data.append(data)

        if all_audio_data:
            combined_audio = np.concatenate(all_audio_data)
            processed_audio = self.trim_silence(combined_audio, samplerate)
            sd.play(processed_audio, samplerate)
            sd.wait()
        else:
            print("No audio data generated.")

    @staticmethod
    def preprocess_text(text):
        text = pre_processors.abbreviations(text)
        text = pre_processors.end_of_line(text)
        text = pre_processors.tone_marks(text)
        return text

    @staticmethod
    def tokenize_and_minimize(text):
        sentences = re.split('(?<=[.!?])\s+', text)
        
        minimized_tokens = []
        for sentence in sentences:
            if len(sentence) <= 100:
                minimized_tokens.append(sentence)
            else:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > 100:
                        if current_chunk:
                            minimized_tokens.append(current_chunk.strip())
                            current_chunk = "<continue> " + word
                        else:
                            minimized_tokens.append(word)
                    else:
                        current_chunk += " " + word

                if current_chunk:
                    minimized_tokens.append(current_chunk.strip())

        return minimized_tokens

    def trim_silence(self, audio, samplerate):
        max_silence_samples = int(self.max_silence_ms * samplerate / 1000)

        is_silent = np.abs(audio) < self.silence_threshold

        silent_regions = np.where(np.diff(is_silent.astype(int)))[0]

        if len(silent_regions) < 2:
            return audio

        processed_chunks = []
        start = 0

        for i in range(0, len(silent_regions) - 1, 2):
            silence_start, silence_end = silent_regions[i], silent_regions[i + 1]

            chunk_start = max(start, silence_start - max_silence_samples)

            chunk_end = min(silence_end, silence_start + max_silence_samples)
            
            processed_chunks.append(audio[chunk_start:chunk_end])
            start = silence_end

        processed_chunks.append(audio[start:])

        return np.concatenate(processed_chunks)



class KyutaiAudio(BaseAudio):
    REQUIRED_PACKAGES = {
        "moshi": "0.2.8",
        "sphn": "0.2.0"
    }
    
    def __init__(self):
        super().__init__()

        from core.utilities import check_and_install_dependencies

        if not check_and_install_dependencies(
            self.REQUIRED_PACKAGES, 
            backend_name="Kyutai"
        ):
            raise RuntimeError("Kyutai dependencies not available")

        self.load_config('config.yaml', 'kyutai')
        self.initialize_device()
        self.initialize_model()

    def create_checkpoint_info_from_cache(self, downloaded_paths):
        from moshi.models.loaders import CheckpointInfo
        import json
        from pathlib import Path

        config_path = Path(downloaded_paths["config.json"])
        with open(config_path, 'r') as f:
            raw_config = json.load(f)

        moshi_weights = Path(downloaded_paths["dsm_tts_1e68beda@240.safetensors"])
        mimi_weights = Path(downloaded_paths["tokenizer-e351c8d8-checkpoint125.safetensors"])
        tokenizer_path = Path(downloaded_paths["tokenizer_spm_8k_en_fr_audio.model"])

        lm_config = dict(raw_config)
        tts_config = lm_config.pop("tts_config", {})
        stt_config = lm_config.pop("stt_config", {})
        model_id = lm_config.pop("model_id", {})
        lm_gen_config = lm_config.pop("lm_gen_config", {})
        model_type = lm_config.pop("model_type", "moshi")

        lm_config.pop("moshi_name", None)
        lm_config.pop("mimi_name", None)
        lm_config.pop("tokenizer_name", None)
        lm_config.pop("lora_name", None)

        return CheckpointInfo(
            moshi_weights=moshi_weights,
            mimi_weights=mimi_weights,
            tokenizer=tokenizer_path,
            lm_config=lm_config,
            raw_config=raw_config,
            model_type=model_type,
            lora_weights=None,
            lm_gen_config=lm_gen_config,
            tts_config=tts_config,
            stt_config=stt_config,
            model_id=model_id,
        )

    def initialize_model(self):
        try:
            from moshi.models.tts import TTSModel
            from moshi.models.loaders import CheckpointInfo
            from huggingface_hub import hf_hub_download

            my_cprint("Loading Kyutai TTS model...", "yellow")

            hf_repo = self.config.get('hf_repo', 'kyutai/tts-1.6b-en_fr')

            required_files = [
                "config.json",
                "dsm_tts_1e68beda@240.safetensors", 
                "tokenizer-e351c8d8-checkpoint125.safetensors",
                "tokenizer_spm_8k_en_fr_audio.model"
            ]

            need_download = False
            for filename in required_files:
                try:
                    hf_hub_download(repo_id=hf_repo, filename=filename, cache_dir=CACHE_DIR, token=False, local_files_only=True)
                except Exception:
                    need_download = True
                    break

            if need_download:
                my_cprint("Downloading Kyutai model files...", "yellow")
            
            downloaded_paths = {}
            for filename in required_files:
                try:
                    file_path = hf_hub_download(
                        repo_id=hf_repo,
                        filename=filename,
                        cache_dir=CACHE_DIR,
                        token=False
                    )
                    downloaded_paths[filename] = file_path
                except Exception as e:
                    my_cprint(f"Error downloading {filename}: {e}", "red")
                    raise

            checkpoint_info = self.create_checkpoint_info_from_cache(downloaded_paths)
            
            self.tts_model = TTSModel.from_checkpoint_info(checkpoint_info, device=torch.device(self.device))

            quality_mapping = {
                "base": 4,
                "medium": 16,
                "high": 32
            }

            quality = self.config.get('quality', 'high')
            self.tts_model.n_q = quality_mapping.get(quality, 32)
            self.tts_model.temp = self.config.get('temp', 0.6)
            self.tts_model.mimi.set_num_codebooks(self.tts_model.n_q)

            my_cprint(f"Kyutai model loaded successfully! (Quality: {quality}, n_q: {self.tts_model.n_q})", "green")

            self.setup_voice_conditioning()

        except Exception as e:
            my_cprint(f"Error initializing Kyutai model: {str(e)}", "red")
            raise

    def setup_voice_conditioning(self):
        try:
            voice_name = self.config.get('voice', 'expresso/ex03-ex01_happy_001_channel1_334s.wav')
            cfg_coef = self.config.get('cfg_coef', 2.0)
            
            voice_path = self.tts_model.get_voice_path(voice_name)
            self.condition_attributes = self.tts_model.make_condition_attributes([voice_path], cfg_coef=cfg_coef)
            
            voice_display = self.config.get('voice_display_name', 'Happy Male')
            my_cprint(f"Voice conditioning loaded: {voice_display}", "green")
            
        except Exception as voice_error:
            my_cprint(f"Voice loading failed: {voice_error}", "yellow")
            my_cprint("Using model without voice conditioning", "yellow")
            self.condition_attributes = self.tts_model.make_condition_attributes([], cfg_coef=None)

    @torch.inference_mode()
    def generate_speech_for_sentence(self, sentence):
        try:
            entries = self.tts_model.prepare_script([sentence], padding_between=1)

            pcms = []
            def on_frame(frame):
                if (frame != -1).all():
                    pcm = self.tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))

            all_entries = [entries]
            all_condition_attributes = [self.condition_attributes]

            with self.tts_model.mimi.streaming(len(all_entries)):
                result = self.tts_model.generate(all_entries, all_condition_attributes, on_frame=on_frame)

            if pcms:
                audio = np.concatenate(pcms, axis=-1)
                return audio
            else:
                return None

        except Exception as e:
            return None

    @torch.inference_mode()
    def process_text_to_audio(self, sentences):
        for sentence in tqdm(sentences, desc="Processing Sentences"):
            if not sentence.strip() or self.stop_event.is_set():
                continue

            try:
                audio = self.generate_speech_for_sentence(sentence.strip())

                if audio is not None:
                    self.audio_queue.put((audio, self.tts_model.mimi.sample_rate))
                else:
                    print("Failed to generate audio for sentence")

            except Exception as e:
                print(f"Error processing sentence: {str(e)}")
                continue

        self.audio_queue.put(None)


def run_tts(config_path, input_text_file):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        tts_model = config.get('tts', {}).get('model', 'bark')

    if tts_model == 'bark':
        audio_class = BarkAudio()
    elif tts_model == 'whisperspeech':
        audio_class = WhisperSpeechAudio()
    elif tts_model == 'chattts':
        audio_class = ChatTTSAudio()
    elif tts_model == 'googletts':
        audio_class = GoogleTTSAudio()
    elif tts_model == 'kokoro':
        audio_class = KokoroAudio()
    elif tts_model == 'chatterbox':
        audio_class = ChatterboxAudio()
    elif tts_model == 'kyutai':
        audio_class = KyutaiAudio()
    else:
        raise ValueError(f"Invalid TTS model specified in config.yaml: {tts_model}")

    audio_class.run(input_text_file)
