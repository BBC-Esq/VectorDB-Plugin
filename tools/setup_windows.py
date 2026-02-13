import os
import subprocess
import sys

cache_dir = os.path.join(
    os.environ.get("USERPROFILE", os.path.expanduser("~")),
    ".triton"
)

if os.path.isdir(cache_dir):
    print(f"\nRemoving Triton cache at {cache_dir} via OS command…")
    subprocess.run(f'rmdir /S /Q "{cache_dir}"', shell=True, check=False)
    print("Triton cache removed.\n")
else:
    print("\nNo Triton cache found to clean.\n")

import subprocess
import time
import tkinter as tk
from tkinter import messagebox
from replace_sourcecode import (
    replace_sentence_transformer_file,
    replace_chattts_file,
    add_cuda_files,
    setup_vector_db,
    check_embedding_model_dimensions,
)

from constants import priority_libs, libs, full_install_libs

start_time = time.time()

def has_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False

python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

hardware_type = "GPU" if has_nvidia_gpu() else "CPU"

def tkinter_message_box(title, message, type="info", yes_no=False):
    root = tk.Tk()
    root.withdraw()
    if yes_no:
        result = messagebox.askyesno(title, message)
    elif type == "error":
        messagebox.showerror(title, message)
        result = False
    else:
        messagebox.showinfo(title, message)
        result = True
    root.destroy()
    return result

def check_python_version_and_confirm():
    major, minor = map(int, sys.version.split()[0].split('.')[:2])
    if major == 3 and minor in [11, 12, 13]:
        return tkinter_message_box(
            "Confirmation",
            f"Python version {sys.version.split()[0]} was detected, which is compatible.\n\nClick YES to proceed or NO to exit.",
            yes_no=True
        )
    else:
        tkinter_message_box(
            "Python Version Error",
            "This program requires Python 3.11, 3.12 or 3.13\n\nPython versions prior to 3.11 or after 3.14 are not yet supported.\n\nExiting the installer...",
            type="error"
        )
        return False

def is_nvidia_gpu_installed():
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def manual_installation_confirmation():
    if not tkinter_message_box("Confirmation", "Have you installed Git?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Git Large File Storage?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Pandoc?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    if not tkinter_message_box("Confirmation", "Have you installed Microsoft Build Tools and/or Visual Studio with the necessary libraries to compile code?\n\nClick YES to confirm or NO to cancel installation.", yes_no=True):
        return False
    return True

if not check_python_version_and_confirm():
    sys.exit(1)

nvidia_gpu_detected = is_nvidia_gpu_installed()
if nvidia_gpu_detected:
    message = "An NVIDIA GPU has been detected.\n\nDo you want to proceed with the installation?"
else:
    message = "No NVIDIA GPU has been detected. An NVIDIA GPU is required for this script to function properly.\n\nDo you still want to proceed with the installation?"

if not tkinter_message_box("GPU Detection", message, yes_no=True):
    sys.exit(1)

if not manual_installation_confirmation():
    sys.exit(1)

def upgrade_pip_setuptools_wheel(max_retries=5, delay=3):
    upgrade_commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "--no-cache-dir"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "wheel", "--no-cache-dir"]
    ]

    for command in upgrade_commands:
        package = command[5]
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1} of {max_retries}: Upgrading {package}...")
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=480)
                print(f"\033[92mSuccessfully upgraded {package}\033[0m")
                break
            except subprocess.CalledProcessError as e:
                print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts.")
            except Exception as e:
                print(f"An unexpected error occurred while upgrading {package}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Failed to upgrade {package} after {max_retries} attempts due to unexpected errors.")

def pip_install(library, with_deps=False, max_retries=5, delay=3):
    pip_args = ["uv", "pip", "install", library]
    if not with_deps:
        pip_args.append("--no-deps")

    for attempt in range(max_retries):
        try:
            print(f"\nAttempt {attempt + 1} of {max_retries}: Installing {library}{' with dependencies' if with_deps else ''}")
            subprocess.run(pip_args, check=True, capture_output=True, text=True, timeout=600)
            print(f"\033[92mSuccessfully installed {library}{' with dependencies' if with_deps else ''}\033[0m")
            return attempt + 1
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed. Error: {e.stderr.strip()}")
            if attempt < max_retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to install {library} after {max_retries} attempts.")
                return 0

def install_libraries(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install(library)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

def install_libraries_with_deps(libraries):
    failed_installations = []
    multiple_attempts = []

    for library in libraries:
        attempts = pip_install(library, with_deps=True)
        if attempts == 0:
            failed_installations.append(library)
        elif attempts > 1:
            multiple_attempts.append((library, attempts))
        time.sleep(0.1)

    return failed_installations, multiple_attempts

# 1. upgrade pip, setuptools, wheel
print("Upgrading pip, setuptools, and wheel:")
upgrade_pip_setuptools_wheel()

# 2. install uv
print("Installing uv:")
subprocess.run(["pip", "install", "uv"], check=True)

# 3. install priority_libs
print("\nInstalling priority libraries:")
try:
    hardware_specific_libs = priority_libs[python_version][hardware_type]

    try:
        common_libs = priority_libs[python_version]["COMMON"]
    except KeyError:
        common_libs = []

    all_priority_libs = hardware_specific_libs + common_libs

    priority_failed, priority_multiple = install_libraries(all_priority_libs)
except KeyError:
    tkinter_message_box("Version Error", f"No libraries configured for Python {python_version} with {hardware_type} configuration", type="error")
    sys.exit(1)

# 4. install libs
print("\nInstalling other libraries:")
other_failed, other_multiple = install_libraries(libs)

# 5. install full_install_libs
print("\nInstalling libraries with dependencies:")
full_install_failed, full_install_multiple = install_libraries_with_deps(full_install_libs)

print("\n----- Installation Summary -----")

all_failed = priority_failed + other_failed + full_install_failed
all_multiple = priority_multiple + other_multiple + full_install_multiple

if all_failed:
    print("\033[91m\nThe following libraries failed to install:\033[0m")
    for lib in all_failed:
        print(f"\033[91m- {lib}\033[0m")

if all_multiple:
    print("\033[93m\nThe following libraries required multiple attempts to install:\033[0m")
    for lib, attempts in all_multiple:
        print(f"\033[93m- {lib} (took {attempts} attempts)\033[0m")

if not all_failed and not all_multiple:
    print("\033[92mAll libraries installed successfully on the first attempt.\033[0m")
elif not all_failed:
    print("\033[92mAll libraries were eventually installed successfully.\033[0m")

if all_failed:
    sys.exit(1)

# 6. clear triton cache
from utilities import clean_triton_cache
clean_triton_cache()

# 7. replace sourcode files
replace_sentence_transformer_file()
replace_chattts_file()
add_cuda_files()
setup_vector_db()
check_embedding_model_dimensions()

# 8. Create directores if needed
def create_directory_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "Models")
    subdirs = ["chat", "tts", "vector", "vision", "whisper"]

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created Models directory: {models_dir}")

    for subdir in subdirs:
        subdir_path = os.path.join(models_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Ensured subdirectory exists: {subdir_path}")

create_directory_structure()

# 9. Update config.yaml with necessary configurations, but only if needed
def update_config_yaml():
    """
    1. If 'created_databases' parent key doesn't exist, creates it
    2. If 'user_manual' mapping doesn't exist under 'created_databases', adds it with:
       - chunk_overlap: 599
       - chunk_size: 1200
       - model: path to the BAAI vector model

    3. If 'openai' parent key doesn't exist, creates it
    4. For each child key under 'openai', adds only if missing:
       - api_key: empty string
       - model: 'gpt-4o-mini'
       - reasoning_effort: 'medium'

    5. If 'server' parent key doesn't exist, creates it
    6. For each child key under 'server', adds only if missing:
       - api_key: empty string
       - connection_str: 'http://localhost:1234/v1'
       - show_thinking: 'medium'

    7. Removes any unauthorized child keys from the 'server' mapping,
       keeping only 'api_key', 'connection_str', and 'show_thinking'

    Existing values are preserved when already present in the configuration.
    """
    import yaml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}

    vector_model_path = os.path.join(script_dir, 'Models', 'vector', 'BAAI--bge-small-en-v1.5')

    if 'created_databases' not in config:
        config['created_databases'] = {}

    config['created_databases']['user_manual'] = {
        'chunk_overlap': 599,
        'chunk_size': 1200,
        'model': vector_model_path
    }

    if 'openai' not in config:
        config['openai'] = {}

    if 'api_key' not in config['openai']:
        config['openai']['api_key'] = ''
    if 'model' not in config['openai']:
        config['openai']['model'] = 'gpt-4o-mini'
    if 'reasoning_effort' not in config['openai']:
        config['openai']['reasoning_effort'] = 'medium'

    if 'server' not in config:
        config['server'] = {}

    if 'api_key' not in config['server']:
        config['server']['api_key'] = ''
    if 'connection_str' not in config['server']:
        config['server']['connection_str'] = 'http://localhost:1234/v1'
    if 'show_thinking' not in config['server']:
        config['server']['show_thinking'] = 'medium'

    server_allowed_keys = {'api_key', 'connection_str', 'show_thinking'}
    server_keys = list(config['server'].keys())
    for key in server_keys:
        if key not in server_allowed_keys:
            del config['server'][key]
    if 'chatterbox' not in config:
        config['chatterbox'] = {
            'device': 'auto'
        }

    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False)

update_config_yaml()

end_time = time.time()
total_time = end_time - start_time
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

print(f"\033[92m\nTotal installation time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\033[0m")
