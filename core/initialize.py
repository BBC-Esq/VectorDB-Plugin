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
        config_data = yaml.safe_load(stream) or {}

    if not config_data.get('Compute_Device'):
        config_data['Compute_Device'] = {}

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
    staging_folder = Path('Vector_DB_restore_tmp')
    previous_folder = Path('Vector_DB_previous_tmp')

    if previous_folder.exists():
        if destination_folder.exists():
            shutil.rmtree(previous_folder)
        else:
            previous_folder.rename(destination_folder)
    if staging_folder.exists():
        shutil.rmtree(staging_folder)

    if not backup_folder.is_dir() or not any(backup_folder.iterdir()):
        raise FileNotFoundError("There is no backup to restore: the Vector_DB_Backup folder is missing or empty.")

    try:
        shutil.copytree(backup_folder, staging_folder)
    except Exception:
        shutil.rmtree(staging_folder, ignore_errors=True)
        raise

    if destination_folder.exists():
        try:
            destination_folder.rename(previous_folder)
        except Exception:
            shutil.rmtree(staging_folder, ignore_errors=True)
            raise

    try:
        staging_folder.rename(destination_folder)
    except Exception:
        if previous_folder.exists():
            previous_folder.rename(destination_folder)
        shutil.rmtree(staging_folder, ignore_errors=True)
        raise

    shutil.rmtree(previous_folder, ignore_errors=True)
    try:
        drop_missing_databases(destination_folder)
    except Exception as e:
        logging.warning(f"The databases were restored, but config.yaml could not be updated to match: {e}")
    logging.info("Successfully restored Vector DB backup.")


def drop_missing_databases(vector_folder):
    from core.utilities import save_config_atomically

    config_path = Path('config.yaml')
    with open(config_path, 'r', encoding='utf-8') as stream:
        config_data = yaml.safe_load(stream) or {}
    entries = config_data.get('created_databases')
    if not isinstance(entries, dict):
        return
    missing = [name for name in entries if name != 'user_manual' and not (vector_folder / str(name)).is_dir()]
    if not missing:
        return
    for name in missing:
        del entries[name]
    database = config_data.get('database')
    if isinstance(database, dict) and database.get('database_to_search') in missing:
        database['database_to_search'] = ''
    save_config_atomically(config_data, config_path, allow_unicode=True)
    print(f"Removed {len(missing)} database(s) from the settings because the backup does not contain them: "
          f"{', '.join(str(name) for name in missing)}")


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
