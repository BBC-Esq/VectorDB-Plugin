import logging
import warnings


def run_tts_in_process(config_path, input_text_file):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    from modules.tts import run_tts
    from core.utilities import my_cprint

    logging.getLogger("transformers").setLevel(logging.ERROR)
    run_tts(config_path, input_text_file)
    my_cprint("TTS models removed from memory.", "red")
