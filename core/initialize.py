import platform
import shutil
from pathlib import Path
import logging

import torch
import yaml
import ctranslate2

from core.constants import PROJECT_ROOT


def get_compute_device_info():
    available_devices = ["cpu"]
    gpu_brand = None
    if torch.cuda.is_available():
        available_devices.append('cuda')

    return {
        'available': available_devices,
        'gpu_brand': gpu_brand
    }

def get_platform_info():
    return {'os': platform.system().lower()}

def get_supported_quantizations(device_type):
    types = ctranslate2.get_supported_compute_types(device_type)
    filtered_types = [q for q in types if q != 'int16']

    desired_order = ['float32', 'float16', 'bfloat16', 'int8_float32', 'int8_float16', 'int8_bfloat16', 'int8']
    return [q for q in desired_order if q in filtered_types]


def update_config_file(**system_info):
    full_config_path = Path('config.yaml').resolve()

    with open(full_config_path, 'r', encoding='utf-8') as stream:
        config_data = yaml.safe_load(stream)

    compute_device_info = system_info.get('Compute_Device', {})
    config_data['Compute_Device']['available'] = compute_device_info.get('available', ['cpu'])

    valid_devices = ['cpu', 'cuda', 'mps']
    for key in ['database_creation', 'database_query']:
        config_data['Compute_Device'][key] = config_data['Compute_Device'].get(key, 'cpu') if config_data['Compute_Device'].get(key) in valid_devices else 'cpu'

    config_data['Supported_CTranslate2_Quantizations'] = {
        'CPU': get_supported_quantizations('cpu'),
        'GPU': get_supported_quantizations('cuda') if torch.cuda.is_available() else []
    }

    for key, value in system_info.items():
        if key not in ('Compute_Device', 'Supported_CTranslate2_Quantizations'):
            config_data[key] = value

    with open(full_config_path, 'w', encoding='utf-8') as stream:
        yaml.safe_dump(config_data, stream)


def check_for_necessary_folders():
    folders = [
        "Assets",
        "Docs_for_DB",
        "Vector_DB_Backup",
        "Vector_DB",
        "Models",
        "Models/vector",
        "Models/chat",
        "Models/tts",
        "Models/vision",
        "Models/whisper",
        "Scraped_Documentation",
    ]
    
    for folder in folders:
        Path(folder).mkdir(exist_ok=True)


def restore_vector_db_backup():
    backup_folder = Path('Vector_DB_Backup')
    destination_folder = Path('Vector_DB')

    if not backup_folder.exists():
        logging.error("Backup folder 'Vector_DB_Backup' does not exist.")
        return

    try:
        if destination_folder.exists():
            shutil.rmtree(destination_folder)
            logging.info("Deleted existing 'Vector_DB' folder.")
        destination_folder.mkdir()
        logging.info("Created 'Vector_DB' folder.")

        for item in backup_folder.iterdir():
            dest_path = destination_folder / item.name
            if item.is_dir():
                shutil.copytree(item, dest_path)
                logging.info(f"Copied directory: {item.name}")
            else:
                shutil.copy2(item, dest_path)
                logging.info(f"Copied file: {item.name}")
        logging.info("Successfully restored Vector DB backup.")
    except Exception as e:
        logging.error(f"Error restoring Vector DB backup: {e}")


def delete_chat_history():
    chat_history_path = PROJECT_ROOT / 'chat_history.txt'
    chat_history_path.unlink(missing_ok=True)


def main():
    compute_device_info = get_compute_device_info()
    platform_info = get_platform_info()
    update_config_file(Compute_Device=compute_device_info, Platform_Info=platform_info)
    check_for_necessary_folders()
    delete_chat_history()

if __name__ == "__main__":
    main()
