from multiprocessing import Process
from pathlib import Path
import warnings
import shutil
import json

import torch
import av
from langchain_community.docstore.document import Document

import whisper_s2t
from whisper_s2t.backends.ctranslate2.hf_utils import download_model
from extract_metadata import extract_audio_metadata
from constants import WHISPER_MODELS

warnings.filterwarnings("ignore")

current_directory = Path(__file__).parent
CACHE_DIR = current_directory / "Models" / "whisper"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class WhisperTranscriber:
    def __init__(self, model_key, batch_size):
        model_info = WHISPER_MODELS[model_key]
        self.model_identifier = model_info['repo_id']
        self.compute_type = model_info['precision']
        self.batch_size = batch_size
        self.cache_dir = str(CACHE_DIR)

        script_dir = Path(__file__).parent
        self.model_dir = script_dir / "Models" / "whisper"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_kwargs = {
            'compute_type': self.compute_type,
            'asr_options': {
                "beam_size": 5,
                "best_of": 1,
                "patience": 2,
                "length_penalty": 1,
                "repetition_penalty": 1.01,
                "no_repeat_ngram_size": 0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.5,
                "prefix": None,
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": True,
                "max_initial_timestamp": 1.0,
                "word_timestamps": False,
                "sampling_temperature": 1.0,
                "return_scores": True,
                "return_no_speech_prob": True,
                "word_aligner_model": 'tiny',
            },
            'model_identifier': self.model_identifier,
            'backend': 'CTranslate2',
        }

        if 'large-v3' in self.model_identifier:
            self.model_kwargs['n_mels'] = 128

    def start_transcription_process(self, audio_file):
        self.audio_file = audio_file
        process = Process(target=self.transcribe_and_create_document)
        process.start()
        process.join()

    @torch.inference_mode()
    def transcribe_and_create_document(self):
        audio_file_str = str(self.audio_file)
        converted_audio_file = self.convert_to_wav(audio_file_str)
        
        try:
            downloaded_path = download_model(
                size_or_id=self.model_identifier,
                cache_dir=str(CACHE_DIR)
            )
            
            model_kwargs = self.model_kwargs.copy()
            model_kwargs.pop('model_identifier', None)
            model_kwargs.pop('cache_dir', None)
            
            model = whisper_s2t.load_model(
                model_identifier=downloaded_path,
                **model_kwargs
            )
            
            transcription = self.transcribe(model, [str(converted_audio_file)])
            self.create_document_object(transcription, audio_file_str)

        except Exception as e:
            print(f"Error during transcription: {e}")
            raise

        finally:
            if converted_audio_file != audio_file_str and Path(converted_audio_file).exists():
                try:
                    Path(converted_audio_file).unlink()
                    print(f"Deleted temporary file: {converted_audio_file}")
                except Exception as e:
                    print(f"Error deleting temporary file {converted_audio_file}: {e}")

    def convert_to_wav(self, audio_file):
        if self.is_correct_format(audio_file):
            print("File is already in the correct format.  No pre-processing is necessary.")
            return str(audio_file)
        
        ffmpeg_available = shutil.which('ffmpeg') is not None
        
        if ffmpeg_available:
            print("FFmpeg detected. Sending the audio file to WhisperS2T for pre-processing and transcription.")
            return str(audio_file)
        else:
            print("FFmpeg not detected. Pre-processing with the av library then sending to WhisperS2T for transcription.")
            output_file = f"{Path(audio_file).stem}_temp_converted.wav"
            output_path = Path(__file__).parent / output_file
            return self.convert_with_av(audio_file, output_path)

    def is_correct_format(self, audio_file):
        try:
            with av.open(audio_file) as container:
                stream = container.streams.audio[0]
                return stream.sample_rate == 16000 and stream.channels == 1 and container.format.name == 'wav'
        except Exception as e:
            print(f"Error checking audio format: {e}")
            return False


    def convert_with_av(self, audio_file, output_path):
        try:
            with av.open(audio_file) as input_container:
                input_stream = input_container.streams.audio[0]
                
                output_container = av.open(str(output_path), mode='w')
                output_stream = output_container.add_stream('pcm_s16le', rate=16000)
                output_stream.channels = 1
                
                resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
                
                for frame in input_container.decode(audio=0):
                    frame.pts = None
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames:
                        for resampled_frame in resampled_frames:
                            for packet in output_stream.encode(resampled_frame):
                                output_container.mux(packet)
                
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
                
                output_container.close()
            
            print("Conversion using av complete.")
            return str(output_path)
        except Exception as e:
            print(f"Error converting file with av library {audio_file}: {e}")
            raise

    def transcribe(self, model, files, lang_codes=['en'], tasks=['transcribe'], initial_prompts=[None]):
        out = model.transcribe_with_vad(files,
                                        lang_codes=lang_codes,
                                        tasks=tasks,
                                        initial_prompts=initial_prompts,
                                        batch_size=self.batch_size)
        transcription = " ".join([_['text'] for _ in out[0]]).strip()
        return transcription

    def create_document_object(self, transcription_text, audio_file_path):
        metadata = extract_audio_metadata(audio_file_path)

        # DEBUG 
        # print("Metadata attributes/fields:")
        # for key, value in metadata.items():
            # print(f"{key}: {value}")

        doc = Document(page_content=transcription_text, metadata=metadata)
        
        script_dir = Path(__file__).parent
        docs_dir = script_dir / "Docs_for_DB"
        docs_dir.mkdir(exist_ok=True)
        
        audio_file_name = Path(audio_file_path).stem
        json_file_path = docs_dir / f"{audio_file_name}.json"
        
        doc_dict = {
            "page_content": transcription_text,
            "metadata": metadata
        }
        
        json_file_path.write_text(json.dumps(doc_dict, indent=4), encoding='utf-8')