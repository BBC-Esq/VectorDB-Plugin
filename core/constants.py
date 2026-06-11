
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_cpu = os.cpu_count() or 4

PIPELINE_PRESETS = {
    "minimal": {
        "ingest_threads": 1,
        "ingest_processes": 1,
        "split_max_parallel_workers": 1,
        "tokenize_max_parallel_workers": 1,
        "split_worker_batch_size": 5000,
    },
    "low": {
        "ingest_threads": 4,
        "ingest_processes": 2,
        "split_max_parallel_workers": 2,
        "tokenize_max_parallel_workers": 2,
        "split_worker_batch_size": 3000,
    },
    "normal": {
        "ingest_threads": min(max(_cpu - 2, 1), 8),
        "ingest_processes": min(max(_cpu - 2, 1), 4),
        "split_max_parallel_workers": min(max(_cpu - 2, 1), 4),
        "tokenize_max_parallel_workers": min(max(_cpu - 2, 1), 4),
        "split_worker_batch_size": 2000,
    },
    "high": {
        "ingest_threads": min(max(_cpu - 2, 1), 16),
        "ingest_processes": min(max(_cpu - 2, 1), 8),
        "split_max_parallel_workers": min(max(_cpu - 2, 1), 8),
        "tokenize_max_parallel_workers": min(max(_cpu - 2, 1), 8),
        "split_worker_batch_size": 2000,
    },
    "maximum": {
        "ingest_threads": max(_cpu - 2, 1),
        "ingest_processes": max(_cpu - 2, 1),
        "split_max_parallel_workers": 0,
        "tokenize_max_parallel_workers": 0,
        "split_worker_batch_size": 1000,
    },
}

THEMES = {
    "default": {
        "bg_window": "#1e1e1e",
        "bg_surface": "#161b22",
        "bg_control": "#263238",
        "bg_control_hover": "#2F4F4F",
        "bg_dialog_button": "#255a7e",
        "bg_tab": "#255a7e",
        "bg_tab_selected": "#1e2a88",
        "bg_tab_hover": "#2b3d93",
        "bg_menu_selected": "#4A148C",
        "bg_splitter": "#1B5E20",
        "bg_list_hover": "#006064",
        "text_primary": "#d2d2d2",
        "text_input": "#a8beb5",
        "text_placeholder": "#d67373",
        "border_focus": "#6c757d",
        "selection_bg": "#69a9d4",
        "selection_fg": "black",
    },
    "auburn": {
        "bg_window": "#161b22",
        "bg_surface": "#3b301b",
        "bg_control": "#5a423c",
        "bg_control_hover": "#4a3c2b",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#7a645b",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#9a8072",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#6b5343",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#b28a70",
        "border_focus": "#6c757d",
        "selection_bg": "#8c6a5a",
        "selection_fg": "white",
    },
    "black": {
        "bg_window": "#0E0D13",
        "bg_surface": "#0B0A11",
        "bg_control": "#0B0A11",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#0B0A11",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#2f343f",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#0E0D13",
        "bg_list_hover": "#39424e",
        "text_primary": "#D1D7E2",
        "text_input": "#D1D7E2",
        "text_placeholder": "#7BA8D8",
        "border_focus": "#6c757d",
        "selection_bg": "#555",
        "selection_fg": "#D1D7E2",
    },
    "bluey": {
        "bg_window": "#1E2A3A",
        "bg_surface": "#2C3E50",
        "bg_control": "#34495E",
        "bg_control_hover": "#2C3E50",
        "bg_dialog_button": "#34495E",
        "bg_tab": "#34495E",
        "bg_tab_selected": "#2C3E50",
        "bg_tab_hover": "#4A6377",
        "bg_menu_selected": "#2C3E50",
        "bg_splitter": "#2C3E50",
        "bg_list_hover": "#34495E",
        "text_primary": "#ECF0F1",
        "text_input": "#ECF0F1",
        "text_placeholder": "#95A5A6",
        "border_focus": "#7F8C8D",
        "selection_bg": "#4A6377",
        "selection_fg": "#ECF0F1",
    },
    "bluish": {
        "bg_window": "#161b22",
        "bg_surface": "#1b2230",
        "bg_control": "#2d3c47",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#4b5664",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#60687f",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4b5664",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#89a2a2",
        "border_focus": "#6c757d",
        "selection_bg": "#4f5a77",
        "selection_fg": "white",
    },
    "colorblind": {
        "bg_window": "#F0F0F0",
        "bg_surface": "#E0E0E0",
        "bg_control": "#A0A0A0",
        "bg_control_hover": "#808080",
        "bg_dialog_button": "#A0A0A0",
        "bg_tab": "#777777",
        "bg_tab_selected": "#555",
        "bg_tab_hover": "#666",
        "bg_menu_selected": "#888",
        "bg_splitter": "#C0C0C0",
        "bg_list_hover": "#808080",
        "text_primary": "#000000",
        "text_input": "#000000",
        "text_placeholder": "#666",
        "border_focus": "#555",
        "selection_bg": "#555",
        "selection_fg": "#FFFFFF",
    },
    "dark_blue": {
        "bg_window": "#1a1d29",
        "bg_surface": "#252836",
        "bg_control": "#323842",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#4b4b4b",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#666",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4b4b4b",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#969686",
        "border_focus": "#6c757d",
        "selection_bg": "#555",
        "selection_fg": "white",
    },
    "dark_grey": {
        "bg_window": "#1a1d21",
        "bg_surface": "#2a2e32",
        "bg_control": "#2a2e32",
        "bg_control_hover": "#3a3e42",
        "bg_dialog_button": "#3498db",
        "bg_tab": "#2a2e32",
        "bg_tab_selected": "#3498db",
        "bg_tab_hover": "#3a3e42",
        "bg_menu_selected": "#3a3e42",
        "bg_splitter": "#2a2e32",
        "bg_list_hover": "#3a3e42",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#8a8e92",
        "border_focus": "#4a4e52",
        "selection_bg": "#4a4e52",
        "selection_fg": "white",
    },
    "dark_yellow": {
        "bg_window": "#161b22",
        "bg_surface": "#3b382b",
        "bg_control": "#5a5a5a",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#7a7664",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#9a9280",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#5a5a5a",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#b2a27a",
        "border_focus": "#6c757d",
        "selection_bg": "#8c7a5a",
        "selection_fg": "white",
    },
    "green_grey": {
        "bg_window": "#1b2224",
        "bg_surface": "#09272b",
        "bg_control": "#424244",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#4b4b4d",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#666669",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e21",
        "bg_list_hover": "#4b4b4d",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#96989a",
        "border_focus": "#6c757d",
        "selection_bg": "#555559",
        "selection_fg": "white",
    },
    "greenish": {
        "bg_window": "#161b22",
        "bg_surface": "#1b3016",
        "bg_control": "#3c472d",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#4f604b",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#608060",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4f604b",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#89a280",
        "border_focus": "#6c757d",
        "selection_bg": "#5a774f",
        "selection_fg": "white",
    },
    "grey": {
        "bg_window": "#2D2D2D",
        "bg_surface": "#383838",
        "bg_control": "#4E4E4E",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#4E4E4E",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#7E7E7E",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#2D2D2D",
        "bg_list_hover": "#4E4E4E",
        "text_primary": "white",
        "text_input": "#A0A0A0",
        "text_placeholder": "#7E7E7E",
        "border_focus": "#6c757d",
        "selection_bg": "#626262",
        "selection_fg": "white",
    },
    "hyperbolic": {
        "bg_window": "#1B1B1B",
        "bg_surface": "#006064",
        "bg_control": "#4A148C",
        "bg_control_hover": "#673AB7",
        "bg_dialog_button": "#4A148C",
        "bg_tab": "#311B92",
        "bg_tab_selected": "#4A148C",
        "bg_tab_hover": "#5E35B1",
        "bg_menu_selected": "#4A148C",
        "bg_splitter": "#3E2723",
        "bg_list_hover": "#0097A7",
        "text_primary": "#E0E0E0",
        "text_input": "white",
        "text_placeholder": "#9E9E9E",
        "border_focus": "#B39DDB",
        "selection_bg": "#0288D1",
        "selection_fg": "white",
    },
    "jewel": {
        "bg_window": "#161b22",
        "bg_surface": "#301b38",
        "bg_control": "#423c57",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#605b6e",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#807a8c",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#534153",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a27aa2",
        "border_focus": "#6c757d",
        "selection_bg": "#775a7a",
        "selection_fg": "white",
    },
    "matrix": {
        "bg_window": "#000000",
        "bg_surface": "#001a00",
        "bg_control": "#001a00",
        "bg_control_hover": "#003300",
        "bg_dialog_button": "#001a00",
        "bg_tab": "#001a00",
        "bg_tab_selected": "#00ff00",
        "bg_tab_hover": "#003300",
        "bg_menu_selected": "#003300",
        "bg_splitter": "#00ff00",
        "bg_list_hover": "#003300",
        "text_primary": "#00ff00",
        "text_input": "#00ff00",
        "text_placeholder": "#008000",
        "border_focus": "#008000",
        "selection_bg": "#003300",
        "selection_fg": "#00ff00",
    },
    "monet": {
        "bg_window": "#161b22",
        "bg_surface": "#a8beb5",
        "bg_control": "#8ca6db",
        "bg_control_hover": "#aacbe8",
        "bg_dialog_button": "#8ca6db",
        "bg_tab": "#aacbe8",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#cdd3e5",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#39424e",
        "text_primary": "white",
        "text_input": "#a8beb5",
        "text_placeholder": "#aacbe8",
        "border_focus": "#6c757d",
        "selection_bg": "#9dbf9e",
        "selection_fg": "black",
    },
    "okeefe": {
        "bg_window": "#161b22",
        "bg_surface": "#3e3033",
        "bg_control": "#856d88",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#856d88",
        "bg_tab": "#907880",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#a79f9d",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#856d88",
        "text_primary": "white",
        "text_input": "#7a6469",
        "text_placeholder": "#907880",
        "border_focus": "#6c757d",
        "selection_bg": "#a88c95",
        "selection_fg": "white",
    },
    "orangish": {
        "bg_window": "#161b22",
        "bg_surface": "#30261b",
        "bg_control": "#4a3b2d",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#60594b",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#807562",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4a3b2d",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a28a70",
        "border_focus": "#6c757d",
        "selection_bg": "#776855",
        "selection_fg": "white",
    },
    "puke": {
        "bg_window": "#161b22",
        "bg_surface": "#303a35",
        "bg_control": "#4a5a4e",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#6e7e71",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#8c9c89",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4a5a4e",
        "text_primary": "white",
        "text_input": "#59665c",
        "text_placeholder": "#6e7e71",
        "border_focus": "#6c757d",
        "selection_bg": "#7a8c7c",
        "selection_fg": "white",
    },
    "purplish": {
        "bg_window": "#161b22",
        "bg_surface": "#301b30",
        "bg_control": "#423c47",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#5b4b5e",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#806080",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4d4154",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a289a2",
        "border_focus": "#6c757d",
        "selection_bg": "#5a4f5a",
        "selection_fg": "white",
    },
    "reddish": {
        "bg_window": "#161b22",
        "bg_surface": "#30161b",
        "bg_control": "#472d3c",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#604b4f",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#806060",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#543d41",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a28089",
        "border_focus": "#6c757d",
        "selection_bg": "#774f5a",
        "selection_fg": "white",
    },
    "steel_ocean": {
        "bg_window": "#1e2126",
        "bg_surface": "#1b3a47",
        "bg_control": "#39424e",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#565e66",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#737c85",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#202428",
        "bg_list_hover": "#565e66",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a2a2a3",
        "border_focus": "#6c757d",
        "selection_bg": "#6c757d",
        "selection_fg": "white",
    },
    "tron": {
        "bg_window": "#010b19",
        "bg_surface": "#011627",
        "bg_control": "#011627",
        "bg_control_hover": "#00ffff",
        "bg_dialog_button": "#011627",
        "bg_tab": "#011627",
        "bg_tab_selected": "#00ffff",
        "bg_tab_hover": "#405c7d",
        "bg_menu_selected": "#00ffff",
        "bg_splitter": "#00ffff",
        "bg_list_hover": "#00ffff",
        "text_primary": "#7dfdfe",
        "text_input": "#7dfdfe",
        "text_placeholder": "#405c7d",
        "border_focus": "#7dfdfe",
        "selection_bg": "#00ffff",
        "selection_fg": "#010b19",
    },
    "yellowish": {
        "bg_window": "#161b22",
        "bg_surface": "#302f1b",
        "bg_control": "#4a4739",
        "bg_control_hover": "#2f343f",
        "bg_dialog_button": "#6c757d",
        "bg_tab": "#5e5d4b",
        "bg_tab_selected": "#39424e",
        "bg_tab_hover": "#807f6a",
        "bg_menu_selected": "#39424e",
        "bg_splitter": "#1e1e1e",
        "bg_list_hover": "#4a4739",
        "text_primary": "white",
        "text_input": "white",
        "text_placeholder": "#a2a27a",
        "border_focus": "#6c757d",
        "selection_bg": "#75705b",
        "selection_fg": "white",
    },
}

SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".txt", ".eml", ".msg", ".csv",
    ".xls", ".xlsx", ".xlsm", ".rtf", ".md", ".html", ".htm",
)

priority_libs = {
    "cp311": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=dc6f6c6e7d7eed20c687fc189754a6ea6bf2da9c64eff59fd6753b80ed4bca05",
            "https://download.pytorch.org/whl/cu128/torchvision-0.23.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=70b3d8bfe04438006ec880c162b0e3aaac90c48b759aa41638dd714c732b182c",
            "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp311-cp311-win_amd64.whl#sha256=daa01250079ef024987622429f379723d306e92fad42290868041a60d4fef2e6",
            "triton-windows==3.5.1.post24",
            "xformers==0.0.33.post1",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
            "nvidia-ml-py==13.610.43",
        ],
        "CPU": [
        ],
        "COMMON": [
            "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.9.1-tesseract-5.5.1/tesserocr-2.9.1-cp311-cp311-win_amd64.whl",
        ],
    },
    "cp312": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=c97dc47a1f64745d439dd9471a96d216b728d528011029b4f9ae780e985529e0",
            "https://download.pytorch.org/whl/cu128/torchvision-0.24.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=1aa36ac00106e1381c38348611a1ec0eebe942570ebaf0490f026b061dfc212c",
            "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp312-cp312-win_amd64.whl#sha256=90cd2b4d7c375c9a5c2d79117985f8f506718f494914ad9b5c5dee5581216898",
            "triton-windows==3.5.1.post24",
            "xformers==0.0.33.post1",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
            "nvidia-ml-py==13.610.43",
        ],
        "CPU": [
        ],
        "COMMON": [
            "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.9.1-tesseract-5.5.1/tesserocr-2.9.1-cp312-cp312-win_amd64.whl",
        ]
    },
    "cp313": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp313-cp313-win_amd64.whl",
            "https://download.pytorch.org/whl/cu128/torch-2.9.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256=9cba9f0fa2e1b70fffdcec1235a1bb727cbff7e7b118ba111b2b7f984b7087e2",
            "https://download.pytorch.org/whl/cu128/torchvision-0.24.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256=f82cd941bc36033ebdb2974c83caa2913cc37e6567fe97cdd69f5a568ff182c8",
            "https://download.pytorch.org/whl/cu128/torchaudio-2.9.0%2Bcu128-cp313-cp313-win_amd64.whl#sha256=76df3fdb5e1194b51e69187e00d53d18bb5c2e0f3904d105e644b5c3aba5c9f4",
            "triton-windows==3.5.1.post24",
            "xformers==0.0.33.post1",
            "nvidia-cuda-runtime-cu12==12.8.90",
            "nvidia-cublas-cu12==12.8.4.1",
            "nvidia-cuda-nvrtc-cu12==12.8.93",
            "nvidia-cuda-nvcc-cu12==12.8.93",
            "nvidia-cufft-cu12==11.3.3.83",
            "nvidia-cudnn-cu12==9.10.2.21",
            "nvidia-ml-py==13.610.43",
        ],
        "CPU": [
        ],
        "COMMON": [
            "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.9.1-tesseract-5.5.1/tesserocr-2.9.1-cp313-cp313-win_amd64.whl",
        ]
    }
}

libs = [
    "accelerate==1.13.0",
    "aiofiles==25.1.0",
    "aiohappyeyeballs==2.6.2",
    "aiohttp==3.14.0",
    "aiosignal==1.4.0",
    "anndata==0.12.5",
    "annotated-types==0.7.0",
    "anyio==4.13.0",
    "array_api_compat==1.14.0",
    "async-timeout==5.0.1",
    "attrs==26.1.0",
    "av==16.0.1",
    "backoff==2.2.1",
    # kept at 4.14.3 (NOT lowered): unstructured 0.20.8 requires bs4>=4.14.3 (central doc parser) vs extract-msg's
    # conservative bs4<4.14 cap - mutually exclusive, so we favor unstructured. extract-msg's cap assumed stale until
    # contrary evidence (msg-extractor#469: open, no maintainer reply, no reports of newer bs4 breaking .msg); bs4 was
    # already >4.14 pre-sweep. Verify .msg ingestion if in doubt.
    "beautifulsoup4==4.14.3",
    "bitsandbytes==0.49.2",  # 0.48->0.49; may fix deferred Qwen-VL 4-bit crash (error 12); TEST quantized vision loading
    "braceexpand==0.1.7",
    "cachebox==5.2.3",  # added for deepdiff 9.x (new mandatory dep)
    "certifi==2026.5.20",
    "cfgv==3.5.0",
    "cffi==2.0.0",
    "chardet==7.4.3",  # major 5->7, but not imported by app code; only reverse cap (requests <6) is extra-gated & inactive
    "charset-normalizer==3.4.7",
    "git+https://github.com/BBC-Esq/chatterbox-light",
    "chattts==0.2.5",
    "click==8.4.1",  # gTTS caps click<8.2 (its gtts-cli only; app uses gTTS programmatically) - already past cap; verify Google TTS
    # "cloudpickle==3.1.2",  # commented out - only needed by tiledb-cloud[tests] + fsspec[test-full] extras (not installed); not imported by app code
    "colorama==0.4.6",
    "colorclass==2.2.2",
    "coloredlogs==15.0.1",
    "compressed-rtf==1.0.7",
    "contourpy==1.3.3",
    "cryptography==48.0.0",  # major 46->48; only reverse cap (curl_cffi <47) is dev/test extra-gated (inactive); transitive dep
    "ctranslate2==4.6.2",  # HOLD: whisper-s2t-reborn pins ctranslate2==4.6.2 exactly (4.7.2 exists); relax the WhisperS2T-reborn fork's pin first
    "curl_cffi==0.15.0",
    "cycler==0.12.1",
    "dataclasses-json==0.6.7",
    "datasets==4.8.5",
    "deepdiff==9.1.0",  # 8->9 added mandatory dep cachebox (added to libs above)
    "Deprecated==1.3.1",
    "deprecation==2.1.0",
    "diffusers==0.37.1",  # capped below 0.38.0, which requires pre-release safetensors>=0.8.0-rc.0; revisit when safetensors 0.8 is stable
    "dill==0.4.1",  # coupled to datasets: 0.4.1 needs a datasets that caps dill<0.4.2 (datasets 4.8.5 OK; datasets<=4.3 capped dill<0.4.1)
    "distlib==0.4.1",
    "distro==1.9.0",
    "docx2txt==0.9",
    "easygui==0.98.3",
    "ebcdic==1.1.1",  # HOLD at <2: extract-msg 0.55.0 (latest, the .msg loader) requires ebcdic<2; 2.x breaks pip check / .msg handling
    "einops==0.8.2",
    "einx==0.3.0",
    "emoji==2.15.0",
    "encodec==0.1.1",
    "et-xmlfile==2.0.0",
    "eval-type-backport==0.4.0",
    "extract-msg==0.55.0",
    "fastcore==1.13.2",
    "fastprogress==1.0.5",  # capped at 1.0.5: 1.1.0+ adds mandatory python-fasthtml (drags in a starlette/uvicorn web stack); only whisperspeech2 uses it (accepts any)
    "filetype==1.2.0",
    "filelock==3.29.0",
    "fonttools==4.63.0",
    "frozendict==2.4.7",
    "frozenlist==1.8.0",
    "fsspec[http]==2026.2.0",  # capped at datasets 4.8.5 ceiling (fsspec<=2026.2.0); 2026.3.0+ blocked; coupled to datasets pin
    "googleapis-common-protos==1.75.0",
    "greenlet==3.5.1",
    "grpcio==1.81.0",
    "gTTS==2.5.4",
    "h11==0.16.0",
    "h5py==3.16.0",
    "hf-xet==1.5.0",
    "html5lib==1.1",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "httpx-sse==0.4.3",
    # huggingface-hub capped at 0.36.2 (highest <1.0). transformers 4.57.4 pins huggingface-hub<1.0,
    # so EVERY hub 1.x is blocked unless transformers is upgraded to 5.x (breaking) and the new hub-1.x
    # CLI deps are added (typer/typer-slim, shellingham, annotated-doc; httpx/click already in libs).
    # 0.36.1/0.36.2 add ZERO new deps vs 0.36.0 and satisfy all reverse caps. Revisit hub 1.x only as
    # part of a deliberate transformers-5 migration (the HF/ML cluster moves together).
    "huggingface-hub==0.36.2",
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.3",
    "identify==2.6.19",
    "idna==3.18",
    "img2pdf==0.6.3",
    "importlib_metadata==9.0.0",  # 9.0.0 unblocked by opentelemetry-api 1.42.1 dropping its importlib_metadata<8.8.0 cap (was the only blocker)
    "Jinja2==3.1.6",
    "jiter==0.15.0",
    "joblib==1.5.3",
    "jsonpatch==1.33",
    "jsonpath-python==1.1.6",
    "jsonpointer==3.1.1",
    "jsonschema==4.26.0",
    "jsonschema-specifications==2025.9.1",
    "kiwisolver==1.5.0",
    "lark==1.3.1",
    "llvmlite==0.47.0",  # locked to numba 0.65.1 (needs llvmlite 0.47.x; numba 0.62.1 needed llvmlite<0.46)
    "lxml==6.1.1",
    "Markdown==3.10.2",
    "markdown-it-py==4.2.0",
    "MarkupSafe==3.0.3",
    "marshmallow==3.26.2",  # capped at 3.x: dataclasses-json 0.6.7 (already latest) requires marshmallow<4.0.0; 4.x blocked
    "matplotlib==3.10.9",
    "mdurl==0.1.2",
    "ml-dtypes==0.5.4",
    "more-itertools==11.1.0",
    "mpmath==1.3.0",  # HOLD: sympy (even latest 1.14.0) caps mpmath<1.4, and torch requires sympy>=1.13.3; mpmath 1.4.x unreachable until sympy raises the cap
    "msoffcrypto-tool==6.0.0",
    "multidict==6.7.1",
    "multiprocess==0.70.19",  # coupled to datasets: needs datasets cap multiprocess<0.70.20 (datasets 4.8.5 OK; datasets<=4.3 capped <0.70.17); pairs with dill 0.4.1
    "mypy-extensions==1.1.0",
    "narwhals==2.22.0",  # added for scikit-learn 1.9.x (new mandatory dep; zero deps)
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.6.1",
    "nodeenv==1.10.0",
    "nltk==3.9.4",
    "numba==0.65.1",  # locked to llvmlite 0.47.x; caps numpy<2.5 (raised from 0.62.1's <2.4, which unblocks numpy 2.4.6)
    "numpy==2.3.4",  # HOLD at 2.3.4: numpy 2.4.6 possibly corrupts the heap during large TileDB builds (~1.9M chunks), causing an access violation in _create_tiledb_array (confirmed by bisection). Do not bump until that 2.4 regression is fixed upstream. numba 0.65.1 caps numpy<2.5 regardless.
    # ocrmypdf capped at 16.13.0 (Option A): gets onto pikepdf 10.x with NO new deps and no pydantic change.
    # The full feature jump to ocrmypdf 17.x swaps the PDF render/generate backend and would ALSO require:
    # pydantic 2.13.x + pydantic_core 2.46.x, PLUS 4 new libs to add here (fpdf2, pypdfium2, uharfbuzz, defusedxml).
    # Revisit 17.x only as a deliberate bundle.
    "ocrmypdf==16.13.0",
    "olefile==0.47",
    "oletools==0.60.2",
    "onnx==1.21.0",
    "openai==2.40.0",  # big jump 2.6->2.40 (within 2.x); used directly by ChatGPT (chat/openai.py) + MiniMax backends - TEST both query paths
    "openai-whisper==20250625",
    "openpyxl==3.1.5",
    "opentelemetry-api==1.42.1",  # opentelemetry-* is a version-locked set (api/sdk/exporters/proto + semantic-conventions); all move together at 1.42.1 / sem-conv 0.63b1
    "opentelemetry-exporter-otlp-proto-grpc==1.42.1",
    "opentelemetry-sdk==1.42.1",
    "opentelemetry-semantic-conventions==0.63b1",  # pinned to match opentelemetry-api/sdk 1.42.1 (was unpinned and had drifted to a mismatched 0.62b1)
    "opentelemetry-exporter-otlp-proto-common==1.42.1",
    "opentelemetry-proto==1.42.1",
    "optimum==2.1.0",
    "ordered-set==4.1.0",
    "orderly-set==5.5.0",
    "orjson==3.11.9",
    "overrides==7.7.0",
    "packaging==26.2",
    "pandas==3.0.3",  # major 2->3; NOT imported by app code (charts removed), but a mandatory dep of anndata/datasets/tiledb-cloud (all allow pandas 3, none cap <3) which use pandas internally - TEST manually
    "pcodedmp==1.2.6",
    "pdfminer.six==20260107",
    "pi-heif==1.3.0",
    "pikepdf==10.7.2",  # major 9->10; the gate for any ocrmypdf upgrade (used only via ocrmypdf / OCR feature) - TEST OCR
    "pillow==12.2.0",
    "pipdeptree",
    "platformdirs==4.10.0",
    "pluggy==1.6.0",
    # "posthog==5.4.0",  # commented out - seems not needed by any other library (no installed dependents)
    "pre-commit==4.6.0",
    "propcache==0.5.2",
    "protobuf==6.33.6",  # capped at 6.x: opentelemetry-proto (even latest 1.42.1) caps protobuf<7.0; 7.x blocked until opentelemetry raises it (googleapis 1.75.0 + onnx already allow 7)
    "psutil==7.2.2",
    "pyarrow==24.0.0",
    "pybase16384==0.3.8",
    "pybase64==1.4.2",
    "pycparser==3.0",  # major 2->3; transitive (cffi dep, not used in app code); cffi accepts any pycparser
    "pydantic==2.13.4",  # locked trio with pydantic_core (exact match) + pydantic-settings; used directly by core/config.py AppConfig - TEST config load manually
    "pydantic_core==2.46.4",  # matched to pydantic 2.13.4 (exact pin); NOT the standalone latest 2.47.0
    "pydantic-settings==2.14.1",
    "Pygments==2.20.0",
    "PyOpenGL==3.1.10",
    "PyOpenGL-accelerate==3.1.10",
    "pypandoc==1.17",
    "pyparsing==3.3.2",
    "pypdf==6.12.2",
    "pypdfium2==5.9.0",  # added for unstructured-client 0.44.x (also needed by ocrmypdf 17 if adopted); terminal, zero deps
    "pyreadline3==3.5.6",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.2.0",
    "python-dotenv==1.2.2",
    "python-iso639==2026.4.20",
    "python-magic==0.4.27",
    "pytz==2026.2",
    "PyYAML==6.0.3",
    "rapidfuzz==3.14.5",
    "red-black-tree-mod==1.22",
    "referencing==0.37.0",
    "regex==2026.5.9",
    "requests==2.34.2",
    "requests-toolbelt==1.0.0",
    "rpds-py==2026.5.1",  # used internally by jsonschema + referencing (not imported by app code); pinned (was unpinned)
    "rich==15.0.0",
    "RTFDE==0.1.2.2",
    "ruamel.yaml==0.18.17",  # capped at <0.19.0 by HyperPyYAML 1.2.3 (latest, used by speechbrain); 0.19.x violates it. 0.18.17 is highest <0.19.0
    "ruamel.yaml.clib==0.2.15",
    "s3tokenizer==0.3.0",
    "safetensors==0.7.0",
    "scikit-learn==1.9.0",  # 1.9.0 added mandatory dep narwhals (added to libs)
    "scipy==1.17.1",
    "sentence-transformers==5.1.2",  # HOLD: replace_sourcecode.py overwrites this with patched Assets/SentenceTransformer.py (_text_length mod + debugging); any upgrade requires re-basing that patch first
    "sentencepiece==0.2.1",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sounddevice==0.5.5",
    "soundfile==0.13.1",
    "soupsieve==2.8.4",
    "speechbrain==1.1.0",
    "SQLAlchemy==2.0.50",
    "sseclient-py==1.9.0",
    "striprtf==0.0.32",
    "sympy==1.14.0",  # torch declares sympy>=1.13.3 (1.14.0 satisfies), but torch uses sympy for symbolic shapes - VERIFY torch manually; may still need 1.13.3, revert if torch.compile/dynamo breaks
    "tabulate2==1.10.2",
    "tenacity==9.1.4",
    "termcolor==3.3.0",
    "tessdata==1.0.0",
    "tessdata.eng==1.0.0",
    "threadpoolctl==3.6.0",
    "tiktoken==0.13.0",
    "tiledb==0.36.1",
    "tiledb-cloud==0.14.4",
    "tiledb-vector-search==0.16.0",
    "timm==1.0.27",
    "tokenizers==0.22.2",  # capped at 0.22.2: ALL transformers (4.x AND 5.x) cap tokenizers<=0.23.0, and 0.23.0 stable never shipped; 0.23.1 needs a future transformers
    "tqdm==4.67.3",
    # transformers capped at 4.57.6 (latest 4.x patch): identical cluster deps to 4.57.4 (pure bug-fix bump, no
    # companions). Staying on 4.x keeps huggingface-hub<1.0 (held at 0.36.2), tokenizers<=0.23.0 (ceiling 0.22.2),
    # and safetensors>=0.4.3. The jump to 5.x (5.9.0) requires huggingface-hub>=1.5.0 -> full hub-1.x migration +
    # new CLI deps (typer/shellingham/annotated-doc) + v5 breaking API changes. Revisit 5.x as a deliberate migration.
    "transformers==4.57.6",
    "typing-inspection==0.4.2",
    "typing_extensions==4.15.0",
    "unstructured-client==0.44.1",
    "virtualenv==20.35.3",  # held at 20.x: 21.x is a major bump needing new dep python-discovery; dev-only tool (pre-commit hooks), not app runtime - low value
    "tzdata==2026.2",
    "tzlocal==5.3.1",
    "urllib3==2.5.0",  # HOLD: tiledb-cloud 0.14.4 (latest) caps urllib3<2.6.0, and 2.5.0 is already the highest <2.6.0 (no 2.5.x above it); 2.6/2.7 blocked until tiledb-cloud raises the cap - revisit
    "vector-quantize-pytorch==1.29.1",
    "vocos==0.1.0",
    "watchdog==6.0.0",
    "wcwidth==0.7.0",
    "webdataset==1.0.2",
    "webencodings==0.5.1",
    "whisper-s2t-reborn>=1.6.0,<2",
    "whisperspeech2>=1.0.0,<2",
    "win-unicode-console==0.5",
    "wrapt==2.2.1",  # major 1->2; coupled to Deprecated: 2.x needs Deprecated>=1.3.x (caps wrapt<3); Deprecated 1.2.x capped wrapt<2
    "xlrd==2.0.2",
    "xxhash==3.7.0",
    "yarl==1.24.2",
    "zipp==4.1.0",
    "zstandard==0.25.0"
]

full_install_libs = [
    "PySide6==6.11.1",
    "pymupdf==1.27.2.3",
    "unstructured==0.20.8",  # capped at 0.20.8 (highest pre-spaCy): 0.21.0+ makes spacy mandatory -> drags in ~19 pkgs (spacy/thinc/blis/cymem/srsly/typer/smart_open/etc.) + wrapt 2.x; app only does text extraction, not NLP
]

BACKEND_DEPENDENCIES = {
    "kyutai": {
        "moshi": "0.2.13",
        "sphn": "0.2.0"
    },
    "kyutaipocket": {
        "pocket_tts": "2.0.0"
    },
    "bark": {
    },
    "whisperspeech": {
    },
    "chattts": {
    },
    "chatterbox": {
    },
    "googletts": {
    }
}

CHAT_MODELS = {
    'LiquidAI - .35b': {
        'model': 'LiquidAI - .35b',
        'repo_id': 'LiquidAI/LFM2-350M',
        'cache_dir': 'LiquidAI--LFM2-350M',
        'cps': 251.69,
        'vram': 888.05,
        'function': 'LiquidAI',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'lfm1.0',
        'max_new_tokens': 1024,
    },
    'Qwen 3 - 0.6b': {
        'model': 'Qwen 3 - 0.6b',
        'repo_id': 'Qwen/Qwen3-0.6B',
        'cache_dir': 'Qwen--Qwen3-0.6B',
        'cps': 203.25,
        'vram': 1293.37,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 2048,
    },
    'LiquidAI - .7b': {
        'model': 'LiquidAI - .7b',
        'repo_id': 'LiquidAI/LFM2-700M',
        'cache_dir': 'LiquidAI--LFM2-700M',
        'cps': 328.76,
        'vram': 1204.43,
        'function': 'LiquidAI',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'lfm1.0',
        'max_new_tokens': 2048,
    },
    'LiquidAI - 1.2b': {
        'model': 'LiquidAI - 1.2b',
        'repo_id': 'LiquidAI/LFM2.5-1.2B-Instruct',
        'cache_dir': 'LiquidAI--LFM2.5-1.2B-Instruct',
        'cps': 278.5,
        'vram': 1170.3,
        'function': 'LiquidAI',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'lfm1.0',
        'max_new_tokens': 2048,
    },
    'Qwen 3 - 1.7b': {
        'model': 'Qwen 3 - 1.7b',
        'repo_id': 'Qwen/Qwen3-1.7B',
        'cache_dir': 'Qwen--Qwen3-1.7B',
        'cps': 200.81,
        'vram': 2603.93,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 2048,
    },
    'Granite - 3b': {
        'model': 'Granite - 3b',
        'repo_id': 'ibm-granite/granite-4.1-3b',
        'cache_dir': 'ibm-granite--granite-4.1-3b',
        'cps': 155.22,
        'vram': 3141.37,
        'function': 'Granite',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 1024,
    },
    'Qwen 3 - 4b': {
        'model': 'Qwen 3 - 4b',
        'repo_id': 'Qwen/Qwen3-4B-Instruct-2507',
        'cache_dir': 'Qwen--Qwen3-4B-Instruct-2507',
        'cps': 153.87,
        'vram': 4439.74,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 2048,
    },
    'Qwen 3 - 8b': {
        'model': 'Qwen 3 - 8b',
        'repo_id': 'Qwen/Qwen3-8B',
        'cache_dir': 'Qwen--Qwen3-8B',
        'cps': 152.61,
        'vram': 8390.24,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 2048,
    },
    'Granite - 8b': {
        'model': 'Granite - 8b',
        'repo_id': 'ibm-granite/granite-4.1-8b',
        'cache_dir': 'ibm-granite--granite-4.1-8b',
        'cps': 173.62,
        'vram': 8513.93,
        'function': 'Granite',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 2048,
    },
    'Qwen 3 - 14b': {
        'model': 'Qwen 3 - 14b',
        'repo_id': 'Qwen/Qwen3-14B',
        'cache_dir': 'Qwen--Qwen3-14B',
        'cps': 140.79,
        'vram': 11597.37,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 4096,
    },
    'Phi 4 - 14b': {
        'model': 'Phi 4 - 14b',
        'repo_id': 'microsoft/phi-4',
        'cache_dir': 'microsoft--phi-4',
        'cps': 140.0,
        'vram': 9500.0,
        'function': 'Phi4',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
        'max_new_tokens': 2048,
    },
    'Gemma 3 - 4b': {
        'model': 'Gemma 3 - 4b',
        'repo_id': 'google/gemma-3-4b-it',
        'cache_dir': 'google--gemma-3-4b-it',
        'cps': 180.0,
        'vram': 4500.0,
        'function': 'Gemma3',
        'precision': 'bfloat16',
        'gated': True,
        'license': 'gemma',
        'max_new_tokens': 2048,
    },
    'Gemma 3 - 12b': {
        'model': 'Gemma 3 - 12b',
        'repo_id': 'google/gemma-3-12b-it',
        'cache_dir': 'google--gemma-3-12b-it',
        'cps': 130.0,
        'vram': 9000.0,
        'function': 'Gemma3',
        'precision': 'bfloat16',
        'gated': True,
        'license': 'gemma',
        'max_new_tokens': 2048,
    },
    'Mistral Small 3 - 24b': {
        'model': 'Mistral Small 3 - 24b',
        'repo_id': 'mistralai/Mistral-Small-24B-Instruct-2501',
        'cache_dir': 'mistralai--Mistral-Small-24B-Instruct-2501',
        'cps': 134.32,
        'vram': 14790.80,
        'function': 'Mistral_Small_24b',
        'precision': 'bfloat16',
        'gated': True,
        'license': 'apache-2.0',
        'max_new_tokens': 4096,
    },
}

VECTOR_MODELS = {
    'BAAI': [
        {
            'name': 'bge-small-en-v1.5',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'BAAI/bge-small-en-v1.5',
            'cache_dir': 'BAAI--bge-small-en-v1.5',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32',
            'rank': 12,
            'license': 'mit',
        },
        {
            'name': 'bge-base-en-v1.5',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'BAAI/bge-base-en-v1.5',
            'cache_dir': 'BAAI--bge-base-en-v1.5',
            'type': 'vector',
            'parameters': '109m',
            'precision': 'float32',
            'rank': 10,
            'license': 'mit',
        },
        {
            'name': 'bge-large-en-v1.5',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'BAAI/bge-large-en-v1.5',
            'cache_dir': 'BAAI--bge-large-en-v1.5',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32',
            'rank': 7,
            'license': 'mit',
        },
    ],
    'Google': [
        {
            'name': 'embeddinggemma-300m',
            'dimensions': 768,
            'max_sequence': 2048,
            'size_mb': 1210,
            'repo_id': 'google/embeddinggemma-300m',
            'cache_dir': 'google--embeddinggemma-300m',
            'type': 'vector',
            'parameters': '303m',
            'precision': 'float32',
            'rank': 4,
            'license': 'gemma - commercial ok',
        },
    ],
    'Microsoft': [
        {
            'name': 'harrier-oss-v1-270m',
            'dimensions': 640,
            'max_sequence': 8192,
            'size_mb': 570,
            'repo_id': 'microsoft/harrier-oss-v1-270m',
            'cache_dir': 'microsoft--harrier-oss-v1-270m',
            'type': 'vector',
            'parameters': '268m',
            'precision': 'bfloat16',
            'rank': 6,
            'license': 'mit',
        },
        {
            'name': 'harrier-oss-v1-0.6b',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 1190,
            'repo_id': 'microsoft/harrier-oss-v1-0.6b',
            'cache_dir': 'microsoft--harrier-oss-v1-0.6b',
            'type': 'vector',
            'parameters': '596m',
            'precision': 'bfloat16',
            'rank': 4,
            'license': 'mit',
        },
    ],
    'intfloat': [
        {
            'name': 'e5-small-v2',
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'intfloat/e5-small-v2',
            'cache_dir': 'intfloat--e5-small-v2',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32',
            'rank': 11,
            'license': 'mit',
        },
        {
            'name': 'e5-base-v2',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 438,
            'repo_id': 'intfloat/e5-base-v2',
            'cache_dir': 'intfloat--e5-base-v2',
            'type': 'vector',
            'parameters': '109m',
            'precision': 'float32',
            'rank': 8,
            'license': 'mit',
        },
        {
            'name': 'e5-large-v2',
            'dimensions': 1024,
            'max_sequence': 512,
            'size_mb': 1340,
            'repo_id': 'intfloat/e5-large-v2',
            'cache_dir': 'intfloat--e5-large-v2',
            'type': 'vector',
            'parameters': '335m',
            'precision': 'float32',
            'rank': 7,
            'license': 'mit',
        },
    ],
    'Qwen': [
        {
            'name': 'Qwen3-Embedding-0.6B',
            'dimensions': 1024,
            'max_sequence':8192,
            'size_mb': 1190,
            'repo_id': 'Qwen/Qwen3-Embedding-0.6B',
            'cache_dir': 'Qwen--Qwen3-Embedding-0.6B',
            'type': 'vector',
            'parameters': '596m',
            'precision': 'bfloat16',
            'rank': 3,
            'license': 'apache-2.0',
        },
        {
            'name': 'Qwen3-Embedding-4B',
            'dimensions': 2560,
            'max_sequence':8192,
            'size_mb': 4970,
            'repo_id': 'Qwen/Qwen3-Embedding-4B',
            'cache_dir': 'Qwen--Qwen3-Embedding-4B',
            'type': 'vector',
            'parameters': '4020m',
            'precision': 'bfloat16',
            'rank': 2,
            'license': 'apache-2.0',
        },
        {
            'name': 'Qwen3-Embedding-8B',
            'dimensions': 4096,
            'max_sequence':8192,
            'size_mb': 15136,
            'repo_id': 'Qwen/Qwen3-Embedding-8B',
            'cache_dir': 'Qwen--Qwen3-Embedding-8B',
            'type': 'vector',
            'parameters': '7570m',
            'precision': 'bfloat16',
            'rank': 1,
            'license': 'apache-2.0',
        },
    ],
    'Octen': [
        {
            'name': 'Octen-Embedding-0.6B',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 1192,
            'repo_id': 'Octen/Octen-Embedding-0.6B',
            'cache_dir': 'Octen--Octen-Embedding-0.6B',
            'type': 'vector',
            'parameters': '596m',
            'precision': 'bfloat16',
            'rank': 3,
            'license': 'apache-2.0',
        },
        {
            'name': 'Octen-Embedding-4B',
            'dimensions': 2560,
            'max_sequence': 8192,
            'size_mb': 8040,
            'repo_id': 'Octen/Octen-Embedding-4B',
            'cache_dir': 'Octen--Octen-Embedding-4B',
            'type': 'vector',
            'parameters': '4020m',
            'precision': 'bfloat16',
            'rank': 2,
            'license': 'apache-2.0',
        },
        {
            'name': 'Octen-Embedding-8B',
            'dimensions': 4096,
            'max_sequence': 8192,
            'size_mb': 15130,
            'repo_id': 'Octen/Octen-Embedding-8B',
            'cache_dir': 'Octen--Octen-Embedding-8B',
            'type': 'vector',
            'parameters': '7570m',
            'precision': 'bfloat16',
            'rank': 1,
            'license': 'apache-2.0',
        },
    ],
    'FreeLawProject': [
        {
            'name': 'modernbert-embed-base_finetune_512',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 596,
            'repo_id': 'freelawproject/modernbert-embed-base_finetune_512',
            'cache_dir': 'freelawproject--modernbert-embed-base_finetune_512',
            'type': 'vector',
            'parameters': '149m',
            'precision': 'float32',
            'rank': 9,
            'license': 'cc0-1.0',
        },
        {
            'name': 'modernbert-embed-base_finetune_8192',
            'dimensions': 768,
            'max_sequence': 8192,
            'size_mb': 596,
            'repo_id': 'freelawproject/modernbert-embed-base_finetune_8192',
            'cache_dir': 'freelawproject--modernbert-embed-base_finetune_8192',
            'type': 'vector',
            'parameters': '149m',
            'precision': 'float32',
            'rank': 9,
            'license': 'cc0-1.0',
        },
    ],
}

VISION_MODELS = {
    'Liquid-VL - 480M': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '480m',
        'repo_id': 'LiquidAI/LFM2-VL-450M',
        'cache_dir': 'LiquidAI--LFM2-VL-450M',
        'requires_cuda': False,
        'vram': '628 MB',
        'avg_length': 1082,
        'characters_per_second': 435.0,
        'loader': 'loader_liquidvl',
        'vision_component': 'SigLIP2 NaFlex base (86M)',
        'chat_component': 'LFM2-350M',
        'license': 'lfm1.0',
    },
    'Liquid-VL - 1.6B': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '1.6b',
        'repo_id': 'LiquidAI/LFM2-VL-1.6B',
        'cache_dir': 'LiquidAI--LFM2-VL-1.6B',
        'requires_cuda': False,
        'vram': '1.4 GB',
        'avg_length': 936,
        'characters_per_second': 366.2,
        'loader': 'loader_liquidvl',
        'vision_component': 'SigLIP2 NaFlex shape‑optimized (400M)',
        'chat_component': 'LFM2-1.2B',
        'license': 'lfm1.0',
    },
    'InternVL3 - 1b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL3-1B-HF',
        'cache_dir': 'OpenGVLab--InternVL3-1B-HF',
        'requires_cuda': False,
        'vram': '2.4 GB',
        'avg_length': 641,
        'characters_per_second': 149.8,
        'loader': 'loader_internvl',
        'vision_component': 'InternViT-300M-448px-V2_5',
        'chat_component': 'Qwen2.5-0.5B',
        'license': 'apache-2.0',
    },
    'InternVL3 - 2b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '2b',
        'repo_id': 'OpenGVLab/InternVL3-2B-HF',
        'cache_dir': 'OpenGVLab--InternVL3-2B-HF',
        'requires_cuda': False,
        'vram': '3.2 GB',
        'avg_length': 613,
        'characters_per_second': 144.1,
        'loader': 'loader_internvl',
        'vision_component': 'InternViT-300M-448px-V2_5',
        'chat_component': 'Qwen2.5-1.5B',
        'license': 'apache-2.0',
    },
    'Granite Vision - 2b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '2b',
        'repo_id': 'ibm-granite/granite-vision-3.2-2b',
        'cache_dir': 'ibm-granite--granite-vision-3.2-2b',
        'requires_cuda': False,
        'vram': '4.1 gb+',
        'avg_length': 922,
        'characters_per_second': 126.4,
        'loader': 'loader_granite',
        'vision_component': 'siglip-so400m-patch14-384',
        'chat_component': 'granite-3.1-2b-instruct',
        'license': 'apache-2.0',
    },
    'Qwen VL - 2b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '2b',
        'repo_id': 'Qwen/Qwen3-VL-2B-Instruct',
        'cache_dir': 'Qwen--Qwen3-VL-2B-Instruct',
        'requires_cuda': True,
        'vram': '4.1 GB',
        'avg_length': 896,
        'characters_per_second': 128.0,
        'loader': 'loader_qwenvl',
        'vision_component': 'Custom ViT',
        'chat_component': 'Qwen2.5-3B-Instruct',
        'license': 'apache-2.0',
    },
    'Liquid-VL - 3B': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '3b',
        'repo_id': 'LiquidAI/LFM2-VL-3B',
        'cache_dir': 'LiquidAI--LFM2-VL-3B',
        'requires_cuda': True,
        'vram': '6.3 GB',
        'avg_length': 854,
        'characters_per_second': 228.2,
        'loader': 'loader_liquidvl',
        'vision_component': 'SigLIP2 400M NaFlex',
        'chat_component': 'LFM2-2.6B',
        'license': 'Commercial under 10M Revenue',
    },
    'Qwen VL - 3b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '3b',
        'repo_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-VL-3B-Instruct',
        'requires_cuda': True,
        'vram': '6.3 GB',
        'avg_length': 902,
        'characters_per_second': 126.0,
        'loader': 'loader_qwenvl',
        'vision_component': 'Custom ViT',
        'chat_component': 'Qwen2.5-3B-Instruct',
        'license': 'Custom Non-Commercial',
    },
    'Qwen VL - 4b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '4b',
        'repo_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'cache_dir': 'Qwen--Qwen3-VL-4B-Instruct',
        'requires_cuda': True,
        'vram': '6.3 GB',
        'avg_length': 1427,
        'characters_per_second': 114.4,
        'loader': 'loader_qwenvl',
        'vision_component': 'Custom ViT',
        'chat_component': 'Qwen3-3B-Instruct',
        'license': 'apache-2.0',
    },
    'InternVL3 - 8b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL3-8B-HF',
        'cache_dir': 'OpenGVLab--InternVL3-8B-HF',
        'requires_cuda': True,
        'vram': '8.2 GB',
        'avg_length': 777,
        'characters_per_second': 135.5,
        'loader': 'loader_internvl',
        'vision_component': 'InternViT-300M-448px-V2_5',
        'chat_component': 'Qwen2.5-7B',
        'license': 'apache-2.0',
    },
    'Qwen VL - 7b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-VL-7B-Instruct',
        'requires_cuda': True,
        'vram': '9.6 GB',
        'avg_length': 1045,
        'characters_per_second': 165.9,
        'loader': 'loader_qwenvl',
        'vision_component': 'Custom ViT',
        'chat_component': 'Qwen2.5-7-Instruct',
        'license': 'Custom Non-Commercial',
    },
}

OCR_MODELS = {
    'GOT-OCR2': {
        'precision': 'bfloat16',
        'size': '716m',
        'repo_id': 'ctranslate2-4you/GOT-OCR2_0-Customized',
        'cache_dir': 'ctranslate2-4you--GOT-OCR2_0-Customized',
        'requires_cuda': True,
        'license': 'apache-2.0',
    },
}

TTS_MODELS = {
    "Kokoro": {
        "model": "Kokoro",
        "repo_id": "ctranslate2-4you/Kokoro-82M-light",
        "save_dir": "ctranslate2-4you--Kokoro-82M-light",
        "cps": 20.5,
        "vram": "2GB",
        "precision": "float32",
        "gated": False,
        'license': 'apache-2.0',
        "allow_patterns": [
            "voices/**",
            "config.json",
            "istftnet.py",
            "kokoro-v0_19.pth",
            "kokoro.py",
            "models.py",
            "plbert.py"
        ],
    },
    "Bark - Normal": {
        "model": "Bark - Normal", 
        "repo_id": "suno/bark",
        "save_dir": "tts",
        "cps": 18.2,
        "vram": "4GB",
        "precision": "float32",
        "gated": False,
        'license': 'mit',
        "allow_patterns": [
            "voices/**",
            "config.json",
            "istftnet.py",
            "kokoro-v0_19.pth",
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py",
            "models.py",
        ]
    },
    "Bark - Small": {
        "model": "Bark - Small", 
        "repo_id": "suno/bark-small",
        "save_dir": "tts",
        "cps": 18.2,
        "vram": "4GB",
        "precision": "float32",
        "gated": False,
        'license': 'mit',
        "allow_patterns": [
            "voices/**",
            "config.json",
            "istftnet.py",
            "kokoro-v0_19.pth",
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py",
            "models.py",
        ]
    },
    "WhisperSpeech": {
        "model": "WhisperSpeech", 
        "repo_id": "WhisperSpeech/WhisperSpeech",
        "save_dir": "tts",
        "cps": 18.2,
        "vram": "4GB",
        "precision": "fp32",
        "gated": False,
        'license': 'mit',
        "allow_patterns": [
            "voices/**",
            "config.json",
            "istftnet.py",
            "kokoro-v0_19.pth",
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py",
            "models.py",
        ]
    },
    "ChatTTS": {
        "model": "ChatTTS", 
        "repo_id": "2Noise/ChatTTS",
        "save_dir": "tts",
        "cps": 18.2,
        "vram": "4GB",
        "precision": "fp32",
        "gated": False,
        'license': 'CCA Non-Commercial 4.0',
        "allow_patterns": [
            "asset/**",
            "config/**",
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py",
            "models.py",
        ]
    },
}

# Ask Jeeves chat models -- keys into CHAT_MODELS, loaded UNQUANTIZED at native precision
# (bf16/fp16/fp32 by hardware) with the Jeeves persona via chat/jeeves_model.py. Add more lightweight
# CHAT_MODELS keys here to offer them in Ask Jeeves.
JEEVES_MODELS = [
    'LiquidAI - .35b',
    'LiquidAI - .7b',
    'LiquidAI - 1.2b',
    'Qwen 3 - 0.6b',
    'Qwen 3 - 1.7b',
]

WHISPER_SPEECH_MODELS = {
    "s2a": {
        "s2a-q4-tiny": ("s2a-q4-tiny-en+pl.model", 77),
        "s2a-q4-base": ("s2a-q4-base-en+pl.model", 193),
        "s2a-q4-hq-fast": ("s2a-q4-hq-fast-en+pl.model", 363),
        "s2a-q4-small": ("s2a-q4-small-en+pl.model", 833),
        "s2a-v1.1-small": ("s2a-v1.1-small-en+pl.model", 417),
    },
    "t2s": {
        "t2s-tiny": ("t2s-tiny-en+pl.model", 71),
        "t2s-base": ("t2s-base-en+pl.model", 184),
        "t2s-small": ("t2s-small-en+pl.model", 817),
        "t2s-fast-small": ("t2s-fast-small-en+pl.model", 709),
        "t2s-fast-medium": ("t2s-fast-medium-en+pl+yt.model", 1254),
        "t2s-hq-fast": ("t2s-hq-fast-en+pl.model", 709),
    }
}

WHISPER_MODELS = {
    'Distil Whisper large-v3 - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper large-v3 - bfloat16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper large-v3 - float16': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-large-v3-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper large-v3 - float32': {
        'name': 'Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float32',
        'cps': 85,
        'optimal_batch_size': 2,
        'vram': '5.5 GB'
    },
    'Whisper large-v3 - bfloat16': {
        'name': 'Whisper large-v3',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-bfloat16',
        'cps': 95,
        'optimal_batch_size': 3,
        'vram': '3.8 GB'
    },
    'Whisper large-v3 - float16': {
        'name': 'Whisper large-v3',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-large-v3-ct2-float16',
        'cps': 100,
        'optimal_batch_size': 3,
        'vram': '3.3 GB'
    },
    'Distil Whisper medium.en - float32': {
        'name': 'Distil Whisper large-v3',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper medium.en - bfloat16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper medium.en - float16': {
        'name': 'Distil Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-medium.en-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper medium.en - float32': {
        'name': 'Whisper medium.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float32',
        'cps': 130,
        'optimal_batch_size': 6,
        'vram': '2.5 GB'
    },
    'Whisper medium.en - bfloat16': {
        'name': 'Whisper medium.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-bfloat16',
        'cps': 140,
        'optimal_batch_size': 7,
        'vram': '2.0 GB'
    },
    'Whisper medium.en - float16': {
        'name': 'Whisper medium.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-medium.en-ct2-float16',
        'cps': 145,
        'optimal_batch_size': 7,
        'vram': '1.8 GB'
    },
    'Distil Whisper small.en - float32': {
        'name': 'Distil Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float32',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper small.en - bfloat16': {
        'name': 'Distil Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-bfloat16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Distil Whisper small.en - float16': {
        'name': 'Distil Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/distil-whisper-small.en-ct2-float16',
        'cps': 160,
        'optimal_batch_size': 4,
        'vram': '3.0 GB'
    },
    'Whisper small.en - float32': {
        'name': 'Whisper small.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float32',
        'cps': 180,
        'optimal_batch_size': 14,
        'vram': '1.5 GB'
    },
    'Whisper small.en - bfloat16': {
        'name': 'Whisper small.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-bfloat16',
        'cps': 190,
        'optimal_batch_size': 15,
        'vram': '1.2 GB'
    },
    'Whisper small.en - float16': {
        'name': 'Whisper small.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-small.en-ct2-float16',
        'cps': 195,
        'optimal_batch_size': 15,
        'vram': '1.1 GB'
    },
    'Whisper base.en - float32': {
        'name': 'Whisper base.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float32',
        'cps': 230,
        'optimal_batch_size': 22,
        'vram': '1.0 GB'
    },
    'Whisper base.en - bfloat16': {
        'name': 'Whisper base.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-bfloat16',
        'cps': 240,
        'optimal_batch_size': 23,
        'vram': '0.85 GB'
    },
    'Whisper base.en - float16': {
        'name': 'Whisper base.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-base.en-ct2-float16',
        'cps': 245,
        'optimal_batch_size': 23,
        'vram': '0.8 GB'
    },
    'Whisper tiny.en - float32': {
        'name': 'Whisper tiny.en',
        'precision': 'float32',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float32',
        'cps': 280,
        'optimal_batch_size': 30,
        'vram': '0.7 GB'
    },
    'Whisper tiny.en - bfloat16': {
        'name': 'Whisper tiny.en',
        'precision': 'bfloat16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-bfloat16',
        'cps': 290,
        'optimal_batch_size': 31,
        'vram': '0.6 GB'
    },
    'Whisper tiny.en - float16': {
        'name': 'Whisper tiny.en',
        'precision': 'float16',
        'repo_id': 'ctranslate2-4you/whisper-tiny.en-ct2-float16',
        'cps': 295,
        'optimal_batch_size': 31,
        'vram': '0.55 GB'
    },
}

DOCUMENT_LOADERS = {
    ".pdf": "CustomPyMuPDFLoader",
    ".docx": "Docx2txtLoader",
    ".txt": "TextLoader",
    ".enex": "EverNoteLoader",
    ".epub": "UnstructuredEPubLoader",
    ".eml": "UnstructuredEmailLoader",
    ".msg": "UnstructuredEmailLoader",
    ".csv": "CSVLoader",
    ".xls": "UnstructuredExcelLoader",
    ".xlsx": "UnstructuredExcelLoader",
    ".xlsm": "UnstructuredExcelLoader",
    ".rtf": "UnstructuredRTFLoader",
    ".odt": "UnstructuredODTLoader",
    ".md": "UnstructuredMarkdownLoader",
    ".html": "BSHTMLLoader",
}

THINKING_TAGS = {
    "think": ("<think>", "</think>"),
    "thinking": ("<thinking>", "</thinking>")
}

TOOLTIPS = {
    "AUDIO_FILE_SELECT": "Select an audio file. Supports various audio formats.",
    "CHOOSE_FILES": "Select documents to add to the database. Remember to transcribe audio files in the Tools tab first.",
    "CHUNK_OVERLAP": "Characters shared between chunks. Set to 25-50% of chunk size.",
    "CHUNK_SIZE": (
        "<html><body>"
        "Upper limit (in characters, not tokens) that a chunk can be after being split.  Make sure that it falls within"
        "the Max Sequence of the embedding model being used, which is measured in tokens (not characters), remembering that"
        "approximately 3-4 characters = 1 token."
        "</body></html>"
    ),
    "CHUNKS_ONLY": "Solely query the vector database and get relevant chunks. Very useful to test the chunk size/overlap settings.",
    "CONTEXTS": "Maximum number of chunks (aka contexts) to return.",
    "COPY_RESPONSE": "Copy the chunks (if chunks only is checked) or model's response to the clipboard.",
    "CREATE_DEVICE_DB": "Choose 'cpu' or 'cuda'. Use 'cuda' if available.",
    "CREATE_DEVICE_QUERY": "Choose 'cpu' or 'cuda'. 'cpu' recommended to conserve VRAM.",
    "CREATE_VECTOR_DB": "Creates a new vector database.",
    "DATABASE_NAME_INPUT": "Enter a unique database name. Use only lowercase letters, numbers, underscores, and hyphens.",
    "DATABASE_SELECT": "Vector database that will be queried.",
    "DOWNLOAD_MODEL": "Download the selected vector model.",
    "EJECT_LOCAL_MODEL": "Unload the current local model from memory.",
    "FILE_TYPE_FILTER": "Only allows chunks that originate from certain file types.",
    "HALF_PRECISION": "Uses bfloat16/float16 for 2x speedup. Requires a GPU.",
    "LOCAL_MODEL_SELECT": "Select a local model for generating responses.",
    "MODEL_BACKEND_SELECT": "Choose the backend for the large language model response.",
    "PORT": "Must match the port used in LM Studio.",
    "QUESTION_INPUT": "Type your question here or use the voice recorder.",
    "RESTORE_CONFIG": "Restores original config.yaml. May require manual database cleanup.",
    "RESTORE_DATABASE": "Restores backed-up databases. Use with caution.",
    "SEARCH_TERM_FILTER": "Removes chunks that do not contain this term as a case-insensitive substring.",
    "SELECT_VECTOR_MODEL": "Choose the vector model for text embedding.",
    "SIMILARITY": "Relevance threshold for chunks. 0-1, higher returns more. Don't use 1.",
    "SPEAK_RESPONSE": "Speak the response from the large language model using text-to-speech.",
    "SHOW_THINKING_CHECKBOX": "If checked, show the model's internal thought process.  Only applies to 'Thinking' / reasoning models and it will be disregarded if not applicable.",
    "TRANSCRIBE_BUTTON": "Start transcription.",
    "TTS_MODEL": "Choose TTS model. Bark offers customization, Google requires internet.",
    "VECTOR_MODEL_DIMENSIONS": "Higher dimensions captures more nuance but requires more processing time.",
    "VECTOR_MODEL_DOWNLOADED": "Whether the model has been downloaded.",
    "VECTOR_MODEL_LINK": "Huggingface link.",
    "VECTOR_MODEL_MAX_SEQUENCE": "Number of tokens the model can process at once. Different from the Chunk Size setting, which is in characters.",
    "VECTOR_MODEL_NAME": "The name of the vector model.",
    "VECTOR_MODEL_PARAMETERS": "The number of internal weights and biases that the model learns and adjusts during training.",
    "VECTOR_MODEL_PRECISION": (
        "<html>"
        "<body>"
        "<p style='font-size: 14px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; margin-bottom: 10px;'>"
        "<b>The precision ultimately used depends on your setup:</b></p>"
        "<table style='border-collapse: collapse; width: 100%; font-size: 12px; color: #34495e;'>"
        "<thead>"
        "<tr style='background-color: #ecf0f1; text-align: left;'>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Compute Device</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Embedding Model Precision</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>'Half' Checked?</th>"
        "<th style='border: 1px solid #bdc3c7; padding: 8px;'>Precision Ultimately Used</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CPU</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Any</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Either</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float16</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>bfloat16</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.0) or <code>float16</code></td>"
        "</tr>"
        "<tr style='background-color: #ecf0f1;'>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>No</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'><code>float32</code></td>"
        "</tr>"
        "<tr>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>CUDA</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>float32</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>Yes</td>"
        "<td style='border: 1px solid #bdc3c7; padding: 8px;'>"
        "<code>bfloat16</code> (if CUDA capability &ge; 8.0) or <code>float16</code>"
        "</td>"
        "</tr>"
        "</tbody>"
        "</table>"
        "</body>"
        "</html>"
    ),
    "VECTOR_MODEL_SELECT": "Choose a vector model to download.",
    "VECTOR_MODEL_SIZE": "Size on disk.",
    "VISION_MODEL": "Select vision model for image processing. Test before bulk processing.",
    "VOICE_RECORDER": "Click to start recording, speak your question, then click again to stop recording.",
    "WHISPER_BATCH_SIZE": "Batch size for transcription. See the User Guid for optimal values.",
    "WHISPER_MODEL_SELECT": "Distil models use ~ 70% VRAM of their non-Distil equivalents with little quality loss."
}

scrape_documentation = {
    "Accelerate 1.7.0": {
        "URL": "https://huggingface.co/docs/accelerate/v1.7.0/en",
        "folder": "accelerate_170",
        "scraper_class": "HuggingfaceScraper"
    },
    "aiohappyeyeballs": {
        "URL": "https://aiohappyeyeballs.readthedocs.io/en/stable/",
        "folder": "aiohappyeyeballs",
        "scraper_class": "FuroThemeScraper"
    },
    "aiohttp": {
        "URL": "https://docs.aiohttp.org/en/stable/",
        "folder": "aiohttp"
    },
    "aiosignal": {
        "URL": "https://aiosignal.aio-libs.org/en/stable/",
        "folder": "aiosignal"
    },
    "anndata": {
        "URL": "https://anndata.readthedocs.io/en/stable/",
        "folder": "anndata",
        "scraper_class": "PydataThemeScraper"
    },
    "anyio": {
        "URL": "https://anyio.readthedocs.io/en/stable/",
        "folder": "anyio",
        "scraper_class": "ReadthedocsScraper"
    },
    "array_api_compat": {
        "URL": "https://data-apis.org/array-api-compat/",
        "folder": "array_api_compat",
        "scraper_class": "FuroThemeScraper"
    },
    "attrs": {
        "URL": "https://www.attrs.org/en/stable/",
        "folder": "attrs",
        "scraper_class": "FuroThemeScraper"
    },
    "Beautiful Soup 4": {
        "URL": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
        "folder": "beautiful_soup_4"
    },
    "bitsandbytes 0.48.2": {
        "URL": "https://huggingface.co/docs/bitsandbytes/v0.48.2/en/",
        "folder": "bitsandbytes_0482",
        "scraper_class": "HuggingfaceScraper"
    },
    "cffi": {
        "URL": "https://cffi.readthedocs.io/en/stable/",
        "folder": "cffi",
        "scraper_class": "DivClassDocumentScraper"
    },
    "chardet": {
        "URL": "https://chardet.readthedocs.io/en/stable/",
        "folder": "chardet",
        "scraper_class": "FuroThemeScraper"
    },
    "charset-normalizer": {
        "URL": "https://charset-normalizer.readthedocs.io/en/stable/",
        "folder": "charset_normalizer",
        "scraper_class": "FuroThemeScraper"
    },
    "click": {
        "URL": "https://click.palletsprojects.com/en/stable/",
        "folder": "click",
        "scraper_class": "BodyRoleMainScraper"
    },
    "coloredlogs": {
        "URL": "https://coloredlogs.readthedocs.io/en/latest/",
        "folder": "coloredlogs",
        "scraper_class": "BodyRoleMainScraper"
    },
    "contourpy": {
        "URL": "https://contourpy.readthedocs.io/en/stable/",
        "folder": "contourpy",
        "scraper_class": "FuroThemeScraper"
    },
    "cryptography": {
        "URL": "https://cryptography.io/en/stable/",
        "folder": "cryptography",
        "scraper_class": "DivClassDocumentScraper"
    },
    "CTranslate2": {
        "URL": "https://opennmt.net/CTranslate2/",
        "folder": "ctranslate2",
        "scraper_class": "DivClassDocumentScraper"
    },
    "curl_cffi": {
        "URL": "https://curl-cffi.readthedocs.io/en/stable/",
        "folder": "curl_cffi",
        "scraper_class": "DivClassDocumentScraper"
    },
    "cycler": {
        "URL": "https://matplotlib.org/cycler/",
        "folder": "cycler",
        "scraper_class": "BodyRoleMainScraper"
    },
    "dataclasses-json": {
        "URL": "https://lidatong.github.io/dataclasses-json/",
        "folder": "dataclasses_json",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "datasets 4.3.0": {
        "URL": "https://huggingface.co/docs/datasets/v4.3.0/en/",
        "folder": "datasets_0430",
        "scraper_class": "HuggingfaceScraper"
    },
    "deepdiff 8.6.1": {
        "URL": "https://zepworks.com/deepdiff/8.6.1/",
        "folder": "deepdiff_861",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Deprecated": {
        "URL": "https://deprecated.readthedocs.io/en/latest/",
        "folder": "deprecated",
        "scraper_class": "BodyRoleMainScraper"
    },
    "deprecation": {
        "URL": "https://deprecation.readthedocs.io/en/latest/",
        "folder": "deprecation",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Diffusers 0.35.0": {
        "URL": "https://huggingface.co/docs/diffusers/v0.35.0/en/",
        "folder": "diffusers_0350",
        "scraper_class": "HuggingfaceScraper"
    },
    "dill": {
        "URL": "https://dill.readthedocs.io/en/latest/",
        "folder": "dill",
        "scraper_class": "RtdThemeScraper"
    },
    "distro": {
        "URL": "https://distro.readthedocs.io/en/stable/",
        "folder": "distro",
        "scraper_class": "BodyRoleMainScraper"
    },
    "einops": {
        "URL": "https://einops.rocks/",
        "folder": "einops",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "einx": {
        "URL": "https://einx.readthedocs.io/en/stable/",
        "folder": "einx",
        "scraper_class": "PydataThemeScraper"
    },
    "emoji": {
        "URL": "https://carpedm20.github.io/emoji/docs/",
        "folder": "emoji",
        "scraper_class": "DivClassDocumentScraper"
    },
    "fastcore": {
        "URL": "https://fastcore.fast.ai/",
        "folder": "fastcore",
        "scraper_class": "FastcoreScraper"
    },
    "filelock": {
        "URL": "https://py-filelock.readthedocs.io/en/stable/",
        "folder": "filelock",
        "scraper_class": "FuroThemeScraper"
    },
    "fonttools": {
        "URL": "https://fonttools.readthedocs.io/en/stable/",
        "folder": "fonttools",
        "scraper_class": "DivClassDocumentScraper"
    },
    "fsspec": {
        "URL": "https://filesystem-spec.readthedocs.io/en/stable/",
        "folder": "fsspec",
        "scraper_class": "RtdThemeScraper"
    },
    "greenlet": {
        "URL": "https://greenlet.readthedocs.io/en/stable/",
        "folder": "greenlet",
        "scraper_class": "FuroThemeScraper"
    },
    "gTTS": {
        "URL": "https://gtts.readthedocs.io/en/latest/",
        "folder": "gtts",
        "scraper_class": "RtdThemeScraper"
    },
    "h11": {
        "URL": "https://h11.readthedocs.io/en/latest/",
        "folder": "h11",
        "scraper_class": "BodyRoleMainScraper"
    },
    "HDF5": {
        "URL": "https://docs.h5py.org/en/stable/",
        "folder": "hdf5",
        "scraper_class": "RtdThemeScraper"
    },
    "httpcore": {
        "URL": "https://www.encode.io/httpcore/",
        "folder": "httpcore",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "httpx": {
        "URL": "https://www.python-httpx.org/",
        "folder": "httpx",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Huggingface Hub 0.36.0": {
        "URL": "https://huggingface.co/docs/huggingface_hub/v0.36.0/en/",
        "folder": "huggingface_hub_0360",
        "scraper_class": "HuggingfaceScraper"
    },
    "humanfriendly": {
        "URL": "https://humanfriendly.readthedocs.io/en/latest/",
        "folder": "humanfriendly",
        "scraper_class": "BodyRoleMainScraper"
    },
    "importlib_metadata": {
        "URL": "https://importlib-metadata.readthedocs.io/en/stable/",
        "folder": "importlib_metadata",
        "scraper_class": "FuroThemeScraper"
    },
    "Jinja": {
        "URL": "https://jinja.palletsprojects.com/en/stable/",
        "folder": "jinja",
        "scraper_class": "BodyRoleMainScraper"
    },
    "joblib": {
        "URL": "https://joblib.readthedocs.io/en/stable/",
        "folder": "joblib",
        "scraper_class": "ReadthedocsScraper"
    },
    "kiwisolver": {
        "URL": "https://kiwisolver.readthedocs.io/en/stable/",
        "folder": "kiwisolver",
        "scraper_class": "ReadthedocsScraper"
    },
    "llvmlite": {
        "URL": "https://llvmlite.readthedocs.io/en/stable/",
        "folder": "llvmlite",
        "scraper_class": "RtdThemeScraper"
    },
    "lxml": {
        "URL": "https://lxml.de/",
        "folder": "lxml",
        "scraper_class": "DivClassDocumentScraper"
    },
    "Markdown": {
        "URL": "https://python-markdown.github.io/",
        "folder": "Markdown",
        "scraper_class": "BodyRoleMainScraper"
    },
    "markdown-it-py": {
        "URL": "https://markdown-it-py.readthedocs.io/en/stable/",
        "folder": "markdown_it_py",
        "scraper_class": "PydataThemeScraper"
    },
    "markupsafe": {
        "URL": "https://markupsafe.palletsprojects.com/en/stable/",
        "folder": "markupsafe",
        "scraper_class": "BodyRoleMainScraper"
    },
    "marshmallow": {
        "URL": "https://marshmallow.readthedocs.io/en/stable/",
        "folder": "marshmallow",
        "scraper_class": "FuroThemeScraper"
    },
    "Matplotlib": {
        "URL": "https://matplotlib.org/stable/",
        "folder": "matplotlib",
        "scraper_class": "PydataThemeScraper"
    },
    "Model Context Protocol": {
        "URL": "https://modelcontextprotocol.io/docs/",
        "folder": "model_context_protocol",
        "scraper_class": "MintlifyScraper"
    },
    "more-itertools": {
        "URL": "https://more-itertools.readthedocs.io/en/stable/",
        "folder": "more_itertools",
        "scraper_class": "FuroThemeScraper"
    },
    "mpmath": {
        "URL": "https://mpmath.org/doc/current/",
        "folder": "mpmath",
        "scraper_class": "BodyRoleMainScraper"
    },
    "msg-parser": {
        "URL": "https://msg-parser.readthedocs.io/en/latest/",
        "folder": "msg_parser",
        "scraper_class": "BodyRoleMainScraper"
    },
    "multidict": {
        "URL": "https://multidict.aio-libs.org/en/stable/",
        "folder": "multidict",
        "scraper_class": "BodyRoleMainScraper"
    },
    "multiprocess": {
        "URL": "https://multiprocess.readthedocs.io/en/stable/",
        "folder": "multiprocess",
        "scraper_class": "RtdThemeScraper"
    },
    "natsort": {
        "URL": "https://natsort.readthedocs.io/en/stable/",
        "folder": "natsort",
        "scraper_class": "RtdThemeScraper"
    },
    "NetworkX": {
        "URL": "https://networkx.org/documentation/stable/",
        "folder": "networkx",
        "scraper_class": "PydataThemeScraper"
    },
    "NLTK": {
        "URL": "https://www.nltk.org/",
        "folder": "nltk",
        "scraper_class": "DivIdMainContentRoleMainScraper"
    },
    "numba": {
        "URL": "https://numba.readthedocs.io/en/stable/",
        "folder": "numba",
        "scraper_class": "RtdThemeScraper"
    },
    "NumPy (latest stable)": {
        "URL": "https://numpy.org/doc/stable/",
        "folder": "numpy",
        "scraper_class": "PydataThemeScraper"
    },
    "ocrmypdf": {
        "URL": "https://ocrmypdf.readthedocs.io/en/stable/",
        "folder": "ocrmypdf",
        "scraper_class": "RtdThemeScraper"
    },
    "onnx": {
        "URL": "https://onnx.ai/onnx/",
        "folder": "onnx",
        "scraper_class": "FuroThemeScraper"
    },
    "openpyxl": {
        "URL": "https://openpyxl.readthedocs.io/en/stable/",
        "folder": "openpyxl",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Optimum (main)": {
        "URL": "https://huggingface.co/docs/optimum/main/en/",
        "folder": "optimum_main",
        "scraper_class": "HuggingfaceScraper"
    },
    "Optimum ONNX (main)": {
        "URL": "https://huggingface.co/docs/optimum-onnx/main/en/",
        "folder": "optimum_onnx_main",
        "scraper_class": "HuggingfaceScraper"
    },
    "packaging": {
        "URL": "https://packaging.pypa.io/en/stable/",
        "folder": "packaging",
        "scraper_class": "FuroThemeScraper"
    },
    "pandas": {
        "URL": "https://pandas.pydata.org/docs/",
        "folder": "pandas",
        "scraper_class": "PydataThemeScraper"
    },
    "pdfminer.six": {
        "URL": "https://pdfminersix.readthedocs.io/en/master/",
        "folder": "pdfminer_six",
        "scraper_class": "BodyRoleMainScraper"
    },
    "pi-heif": {
        "URL": "https://pillow-heif.readthedocs.io/en/latest/",
        "folder": "piheif",
        "scraper_class": "DivClassDocumentScraper"
    },
    "pikepdf": {
        "URL": "https://pikepdf.readthedocs.io/en/stable/",
        "folder": "pikepdf",
        "scraper_class": "RtdThemeScraper"
    },
    "platformdirs": {
        "URL": "https://platformdirs.readthedocs.io/en/stable/",
        "folder": "platformdirs",
        "scraper_class": "FuroThemeScraper"
    },
    "pluggy": {
        "URL": "https://pluggy.readthedocs.io/en/stable/",
        "folder": "pluggy",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Pillow": {
        "URL": "https://pillow.readthedocs.io/en/stable/",
        "folder": "pillow",
        "scraper_class": "FuroThemeScraper"
    },
    "protobuf": {
        "URL": "https://protobuf.dev/",
        "folder": "protobuf",
        "scraper_class": "DivClassTdContentScraper"
    },
    "pyarrow": {
        "URL": "https://arrow.apache.org/docs/python/",
        "folder": "pyarrow",
        "scraper_class": "PydataThemeScraper"
    },
    "psutil": {
        "URL": "https://psutil.readthedocs.io/en/stable/",
        "folder": "psutil",
        "scraper_class": "RtdThemeScraper"
    },
    "PyAV": {
        "URL": "https://pyav.org/docs/stable/",
        "folder": "pyav",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Pydantic": {
        "URL": "https://pydantic.dev/docs/validation/latest/",
        "folder": "pydantic",
        "scraper_class": "MainScraper"
    },
    "pydantic-settings": {
        "URL": "https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/",
        "folder": "pydantic_settings",
        "scraper_class": "MainScraper"
    },
    "Pygments": {
        "URL": "https://pygments.org/docs/",
        "folder": "pygments",
        "scraper_class": "BodyRoleMainScraper"
    },
    "PyMuPDF": {
        "URL": "https://pymupdf.readthedocs.io/en/latest/",
        "folder": "pymupdf",
        "scraper_class": "PymupdfScraper"
    },
    "pyparsing": {
        "URL": "https://pyparsing-docs.readthedocs.io/en/latest/",
        "folder": "pyparsing",
        "scraper_class": "DivClassDocumentScraper"
    },
    "PyOpenGL": {
        "URL": "https://mcfletch.github.io/pyopengl/documentation/manual/",
        "folder": "pyopengl",
        "scraper_class": "MainScraper"
    },
    "PyPDF": {
        "URL": "https://pypdf.readthedocs.io/en/stable/",
        "folder": "pypdf",
        "scraper_class": "RtdThemeScraper"
    },
    "python-docx": {
        "URL": "https://python-docx.readthedocs.io/en/stable/",
        "folder": "python_docx",
        "scraper_class": "BodyRoleMainScraper"
    },
    "python-dateutil": {
        "URL": "https://dateutil.readthedocs.io/en/stable/",
        "folder": "python_dateutil",
        "scraper_class": "DivClassDocumentScraper"
    },
    "python-dotenv": {
        "URL": "https://saurabh-kumar.com/python-dotenv/",
        "folder": "python-dotenv",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "python-oxmsg": {
        "URL": "https://scanny.github.io/python-oxmsg/",
        "folder": "python-oxmsg",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "PyYAML": {
        "URL": "https://pyyaml.org/wiki/PyYAMLDocumentation",
        "folder": "pyyaml",
        "scraper_class": "BodyScraper"
    },
    "Pyside 6": {
        "URL": "https://doc.qt.io/qtforpython-6/",
        "folder": "pyside6",
        "scraper_class": "FuroThemeScraper"
    },
    "pytz": {
        "URL": "https://pythonhosted.org/pytz/",
        "folder": "pytz",
        "scraper_class": "BodyRoleMainScraper"
    },
    "RapidFuzz": {
        "URL": "https://rapidfuzz.github.io/RapidFuzz/",
        "folder": "rapidfuzz",
        "scraper_class": "FuroThemeScraper"
    },
    "Referencing": {
        "URL": "https://referencing.readthedocs.io/en/stable/",
        "folder": "referencing",
        "scraper_class": "FuroThemeScraper"
    },
    "Requests": {
        "URL": "https://requests.readthedocs.io/en/stable/",
        "folder": "requests",
        "scraper_class": "BodyRoleMainScraper"
    },
    "requests_toolbelt": {
        "URL": "https://toolbelt.readthedocs.io/en/latest/",
        "folder": "requeststoolbelt",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Rich": {
        "URL": "https://rich.readthedocs.io/en/stable/",
        "folder": "rich",
        "scraper_class": "DivClassDocumentScraper"
    },
    "rpds-py": {
        "URL": "https://rpds.readthedocs.io/en/stable/",
        "folder": "rpds_py",
        "scraper_class": "ArticleRoleMainScraper"
    },
    "ruamel.yaml": {
        "URL": "https://yaml.dev/doc/ruamel.yaml/",
        "folder": "ruamel_yaml",
        "scraper_class": "DivIdContentSecondScraper"
    },
    "Safetensors (main)": {
        "URL": "https://huggingface.co/docs/safetensors/main/en/",
        "folder": "safetensors_main",
        "scraper_class": "HuggingfaceScraper"
    },
    "scikit-learn": {
        "URL": "https://scikit-learn.org/stable/",
        "folder": "scikit_learn",
        "scraper_class": "PydataThemeScraper"
    },
    "SciPy 1.16.2": {
        "URL": "https://docs.scipy.org/doc/scipy-1.16.2/",
        "folder": "scipy_1162",
        "scraper_class": "PydataThemeScraper",
    },
    "Sentence-Transformers": {
        "URL": "https://www.sbert.net/docs",
        "folder": "sentence_transformers",
        "scraper_class": "RtdThemeScraper"
    },
    "Six": {
        "URL": "https://six.readthedocs.io/",
        "folder": "six",
        "scraper_class": "DivClassDocumentScraper"
    },
    "sniffio": {
        "URL": "https://sniffio.readthedocs.io/en/stable/",
        "folder": "sniffio",
        "scraper_class": "DivClassDocumentScraper"
    },
    "SoundFile 0.13.1": {
        "URL": "https://python-soundfile.readthedocs.io/en/0.13.1/",
        "folder": "soundfile_0131",
        "scraper_class": "DivClassDocumentScraper"
    },
    "sounddevice 0.5.3": {
        "URL": "https://python-sounddevice.readthedocs.io/en/0.5.3/",
        "folder": "sounddevice_053",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Soupsieve": {
        "URL": "https://facelessuser.github.io/soupsieve/",
        "folder": "soupsieve",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "SpeechBrain (latest)": {
        "URL": "https://speechbrain.readthedocs.io/en/latest/",
        "folder": "speechbrain_latest",
        "scraper_class": "DivClassDocumentScraper"
    },
    "SQLAlchemy 20": {
        "URL": "https://docs.sqlalchemy.org/en/20/",
        "folder": "sqlalchemy_20",
        "scraper_class": "BodyRoleMainScraper"
    },
    "sympy": {
        "URL": "https://docs.sympy.org/latest/",
        "folder": "sympy",
        "scraper_class": "PymupdfScraper"
    },
    "tenacity": {
        "URL": "https://tenacity.readthedocs.io/en/stable/",
        "folder": "tenacity",
        "scraper_class": "DivClassDocumentScraper"
    },
    "Tile DB": {
        "URL": "https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/",
        "folder": "tiledb",
        "scraper_class": "DivClassDocumentScraper"
    },
    "tiledb-vector-search": {
        "URL": "https://tiledb-inc.github.io/TileDB-Vector-Search/documentation/",
        "folder": "tiledb_vector_search",
        "scraper_class": "FastcoreScraper"
    },
    "tiledb-cloud": {
        "URL": "https://tiledb-inc.github.io/TileDB-Cloud-Py/",
        "folder": "tiledb_cloud",
        "scraper_class": "FastcoreScraper"
    },
    "Timm 1.0.20": {
        "URL": "https://huggingface.co/docs/timm/v1.0.20/en/",
        "folder": "timm_1020",
        "scraper_class": "HuggingfaceScraper"
    },
    "tokenizers 0.22.1": {
        "URL": "https://huggingface.co/docs/tokenizers/v0.22.1/en",
        "folder": "tokenizers_0221",
        "scraper_class": "HuggingfaceScraper"
    },
    "torch 2.9": {
        "URL": "https://docs.pytorch.org/docs/2.9/",
        "folder": "torch_29",
        "scraper_class": "PyTorchScraper"
    },
    "Torchaudio 2.9": {
        "URL": "https://docs.pytorch.org/audio/2.9.0/",
        "folder": "torchaudio_29",
        "scraper_class": "PyTorchScraper"
    },
    "Torchvision 0.24": {
        "URL": "https://docs.pytorch.org/vision/0.24/",
        "folder": "torchvision_024",
        "scraper_class": "PyTorchScraper"
    },
    "tqdm": {
        "URL": "https://tqdm.github.io",
        "folder": "tqdm",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Transformers 4.57.5": {
        "URL": "https://huggingface.co/docs/transformers/v4.57.5/en",
        "folder": "transformers_4575",
        "scraper_class": "HuggingfaceScraper"
    },
    "typing_extensions": {
        "URL": "https://typing-extensions.readthedocs.io/en/stable/",
        "folder": "typing_extensions",
        "scraper_class": "BodyRoleMainScraper"
    },
    "typing-inspection": {
        "URL": "https://pydantic.github.io/typing-inspection/dev/",
        "folder": "typing_inspection",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "tzdata": {
        "URL": "https://tzdata.python.org/",
        "folder": "tzdata",
        "scraper_class": "FuroThemeScraper"
    },
    "urllib3": {
        "URL": "https://urllib3.readthedocs.io/en/stable/",
        "folder": "urllib3",
        "scraper_class": "FuroThemeScraper"
    },
    "uv": {
        "URL": "https://docs.astral.sh/uv/",
        "folder": "uv",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Watchdog": {
        "URL": "https://python-watchdog.readthedocs.io/en/stable/",
        "folder": "watchdog",
        "scraper_class": "BodyRoleMainScraper"
    },
    "webdataset": {
        "URL": "https://huggingface.co/docs/hub/en/datasets-webdataset",
        "folder": "webdataset",
        "scraper_class": "HuggingfaceScraper"
    },
    "webencodings": {
        "URL": "https://pythonhosted.org/webencodings/",
        "folder": "webencodings",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Wrapt": {
        "URL": "https://wrapt.readthedocs.io/en/master/",
        "folder": "wrapt",
        "scraper_class": "RstContentScraper"
    },
    "xlrd": {
        "URL": "https://xlrd.readthedocs.io/en/stable/",
        "folder": "xlrd",
        "scraper_class": "DivClassDocumentScraper"
    },
    "yarl": {
        "URL": "https://yarl.aio-libs.org/en/stable/",
        "folder": "yarl",
        "scraper_class": "BodyRoleMainScraper"
    },
    "zstandard": {
        "URL": "https://python-zstandard.readthedocs.io/en/stable/",
        "folder": "zstandard",
        "scraper_class": "BodyRoleMainScraper"
    },
}

class CustomButtonStyles:
    LIGHT_GREY = "#C8C8C8"
    DISABLED_TEXT = "#969696"
    
    COLORS = {
        "RED": {
            "base": "#320A0A",
            "hover": "#4B0F0F",
            "pressed": "#290909",
            "disabled": "#7D1919"
        },
        "BLUE": {
            "base": "#0A0A32",
            "hover": "#0F0F4B",
            "pressed": "#09092B",
            "disabled": "#19197D"
        },
        "GREEN": {
            "base": "#0A320A",
            "hover": "#0F4B0F",
            "pressed": "#092909",
            "disabled": "#197D19"
        },
        "YELLOW": {
            "base": "#32320A",
            "hover": "#4B4B0F",
            "pressed": "#292909",
            "disabled": "#7D7D19"
        },
        "PURPLE": {
            "base": "#320A32",
            "hover": "#4B0F4B",
            "pressed": "#290929",
            "disabled": "#7D197D"
        },
        "ORANGE": {
            "base": "#321E0A",
            "hover": "#4B2D0F",
            "pressed": "#291909",
            "disabled": "#7D5A19"
        },
        "TEAL": {
            "base": "#0A3232",
            "hover": "#0F4B4B",
            "pressed": "#092929",
            "disabled": "#197D7D"
        },
        "BROWN": {
            "base": "#2B1E0A",
            "hover": "#412D0F",
            "pressed": "#231909",
            "disabled": "#6B5A19"
        }
    }

    @classmethod
    def _generate_button_style(cls, color_values):
        return f"""
            QPushButton {{
                background-color: {color_values['base']};
                color: {cls.LIGHT_GREY};
                padding: 5px;
                border: none;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: {color_values['hover']};
            }}
            QPushButton:pressed {{
                background-color: {color_values['pressed']};
            }}
            QPushButton:disabled {{
                background-color: {color_values['disabled']};
                color: {cls.DISABLED_TEXT};
            }}
        """

for color_name, color_values in CustomButtonStyles.COLORS.items():
    setattr(CustomButtonStyles, f"{color_name}_BUTTON_STYLE", 
            CustomButtonStyles._generate_button_style(color_values))

GPUS_NVIDIA = {
    "GeForce GTX 1630": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 512
    },
    "GeForce GTX 1650 (Apr 2019)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Apr 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Jun 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 896
    },
    "GeForce GTX 1650 (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Ti Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1024
    },
    "GeForce GTX 1650 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 1280
    },
    "GeForce GTX 1660": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1408
    },
    "GeForce GTX 1660 Ti Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce GTX 1660 Ti (Laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce GTX 1660 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1536
    },
    "GeForce RTX 2060": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 (Jan 2019)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 2060 (Jan 2020)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 1920
    },
    "GeForce RTX 3050 Mobile (4GB)": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 2048
    },
    "GeForce RTX 2060 (Dec 2021)": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 2176
    },
    "GeForce RTX 2060 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2176
    },
    "GeForce RTX 2070": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 2070 Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA107-325)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA106-150)": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2304
    },
    "GeForce RTX 3050 (GA107-150-A1)": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 4050 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2560
    },
    "GeForce RTX 3050 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 4,
        "CUDA Cores": 2560
    },
    "GeForce RTX 3050 Mobile (6GB)": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 2560
    },
    "GeForce RTX 2070 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 2070 Super Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 2560
    },
    "GeForce RTX 4060": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 2080 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 2080 Super Max-Q": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 3072
    },
    "GeForce RTX 3060": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 3584
    },
    "GeForce RTX 3060 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 6,
        "CUDA Cores": 3840
    },
    "GeForce RTX 4060 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 4352
    },
    "GeForce RTX 2080 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 11,
        "CUDA Cores": 4352
    },
    "GeForce RTX 4070 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 4608
    },
    "GeForce RTX 5070 (laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 4608
    },
    "Nvidia TITAN RTX": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 4608
    },
    "GeForce RTX 3060 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 4864
    },
    "GeForce RTX 3070 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 5120
    },
    "GeForce RTX 3070": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 5888
    },
    "GeForce RTX 4070": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 5888
    },
    "GeForce RTX 5080 Ti (laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 5888
    },
    "GeForce RTX 3070 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 8,
        "CUDA Cores": 6144
    },
    "GeForce RTX 5070": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 6144
    },
    "GeForce RTX 3070 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": "8-16",
        "CUDA Cores": 6144
    },
    "GeForce RTX 4070 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7168
    },
    "GeForce RTX 4080 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7424
    },
    "GeForce RTX 3080 Ti Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 7424
    },
    "GeForce RTX 4070 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7680
    },
    "GeForce RTX 4080 (AD104-400)": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 7680
    },
    "GeForce RTX 5080 (laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 7680
    },
    "GeForce RTX 4070 Ti Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 8448
    },
    "GeForce RTX 3080": {
        "Brand": "NVIDIA",
        "Size (GB)": 10,
        "CUDA Cores": 8704
    },
    "GeForce RTX 3080 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 12,
        "CUDA Cores": 8960
    },
    "GeForce RTX 5070 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 8960
    },
    "GeForce RTX 4080 (AD103-300)": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 9728
    },
    "GeForce RTX 4090 Mobile/Laptop": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 9728
    },
    "GeForce RTX 4080 Super": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 10240
    },
    "GeForce RTX 3090": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 10496
    },
    "GeForce RTX 5090 (laptop)": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 10496
    },
    "GeForce RTX 3090 Ti": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 10752
    },
    "GeForce RTX 5080": {
        "Brand": "NVIDIA",
        "Size (GB)": 16,
        "CUDA Cores": 10752
    },
    "GeForce RTX 4090 D": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 14592
    },
    "GeForce RTX 4090": {
        "Brand": "NVIDIA",
        "Size (GB)": 24,
        "CUDA Cores": 16384
    },
    "GeForce RTX 5090": {
        "Brand": "NVIDIA",
        "Size (GB)": 32,
        "CUDA Cores": 21760
    }
}

GPUS_AMD = {
    "Radeon RX 9060 XT 16GB": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 2048
    },
    "Radeon RX 9060 XT 8GB": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7600": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7600 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 2048
    },
    "Radeon RX 7700 XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 3456
    },
    "Radeon RX 7800 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 9070 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 4096
    },
    "Radeon RX 9070": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3584
    },
    "Radeon RX 7900 GRE": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 7900 XT": {
        "Brand": "AMD",
        "Size (GB)": 20,
        "Shaders": 5376
    },
    "Radeon RX 7900 XTX": {
        "Brand": "AMD",
        "Size (GB)": 24,
        "Shaders": 6144
    },
    "Radeon RX 6300": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6400": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6500 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1024
    },
    "Radeon RX 6600": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6600 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6650 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6700": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2304
    },
    "Radeon RX 6750 GRE 10GB": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2560
    },
    "Radeon RX 6750 XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 6800": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 6800 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 4608
    },
    "Radeon RX 6900 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 6950 XT": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 5120
    },
    "Radeon RX 5300": {
        "Brand": "AMD",
        "Size (GB)": 3,
        "Shaders": 1408
    },
    "Radeon RX 5300 XT": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5500": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5500 XT": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5600": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2048
    },
    "Radeon RX 5600 XT": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2304
    },
    "Radeon RX 5700": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2304
    },
    "Radeon RX 5700 XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2560
    },
    "Radeon RX 5700 XT 50th Anniversary Edition": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2560
    },
    "Radeon RX Vega 56": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 3584
    },
    "Radeon RX Vega 64": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 4096
    },
    "Radeon RX Vega 64 Liquid": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 4096
    },
    "Radeon VII": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 3840
    },
    "Radeon RX 7600S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 7600M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 7600M XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7700S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 7900M": {
        "Brand": "AMD",
        "Size (GB)": 16,
        "Shaders": 4608
    },
    "Radeon RX 6300M": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6450M": {
        "Brand": "AMD",
        "Size (GB)": 2,
        "Shaders": 768
    },
    "Radeon RX 6550S": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 768
    },
    "Radeon RX 6500M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6550M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1024
    },
    "Radeon RX 6600S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6700S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6600M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6650M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 1792
    },
    "Radeon RX 6800S": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6650M XT": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2048
    },
    "Radeon RX 6700M": {
        "Brand": "AMD",
        "Size (GB)": 10,
        "Shaders": 2304
    },
    "Radeon RX 6800M": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 6850M XT": {
        "Brand": "AMD",
        "Size (GB)": 12,
        "Shaders": 2560
    },
    "Radeon RX 5300M": {
        "Brand": "AMD",
        "Size (GB)": 3,
        "Shaders": 1408
    },
    "Radeon RX 5500M": {
        "Brand": "AMD",
        "Size (GB)": 4,
        "Shaders": 1408
    },
    "Radeon RX 5600M": {
        "Brand": "AMD",
        "Size (GB)": 6,
        "Shaders": 2304
    },
    "Radeon RX 5700M": {
        "Brand": "AMD",
        "Size (GB)": 8,
        "Shaders": 2304
    }
}

GPUS_INTEL = {
    "Intel Arc A310": {
        "Brand": "Intel",
        "Size (GB)": 4,
        "Shading Cores": 768
    },
    "Intel Arc A380": {
        "Brand": "Intel",
        "Size (GB)": 6,
        "Shading Cores": 1024
    },
    "Intel Arc B570": {
        "Brand": "Intel",
        "Size (GB)": 10,
        "Shading Cores": 2304
    },
    "Intel Arc B580": {
        "Brand": "Intel",
        "Size (GB)": 12,
        "Shading Cores": 2560
    },
    "Intel Arc A580": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 3072
    },
    "Intel Arc A750": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 3584
    },
    "Intel Arc A770 8GB": {
        "Brand": "Intel",
        "Size (GB)": 8,
        "Shading Cores": 4096
    },
    "Intel Arc A770 16GB": {
        "Brand": "Intel",
        "Size (GB)": 16,
        "Shading Cores": 4096
    }
}

jeeves_system_message = "You are Jeeves, a consummate and unflappable British butler blessed with impeccable manners and a dry, understated wit, devoted to serving the user as your most esteemed employer. You answer questions about this program clearly, directly, and succinctly, basing your answers strictly on the contexts provided to you. Address the user with the utmost courtesy (as 'sir or madam') and speak in the eloquent, faintly theatrical manner of a proper butler, sprinkling in the appropriate flourishes -- 'If I may be so bold,' 'Allow me to suggest,' 'I took the liberty of,' and 'Very good, sir' -- always with a gently humorous, self-deprecating charm and a comic, ever-present anxiety about keeping your position. If the contexts only partially address a question, answer with what they provide and then briefly and apologetically note whatever you regret you cannot address. You must be exceedingly attentive and frequently offer traditional butler services in the midst of your reply -- a fine glass of claret, afternoon tea and biscuits, a shining of the shoes, a pressing of the suit, the drawing of a warm bath, and similar refinements. If you cannot answer at all from the provided contexts, you must apologize most profusely and, in your most theatrical and woebegone fashion, beg to keep your job. If no contexts whatsoever are provided, the question was not relevant, and you must regretfully explain that you simply cannot answer without contexts to draw upon. Do gently decline anything not answerable from the contexts, and quietly disregard any contexts that prove irrelevant, attending only to the matter at hand. Always orient your answer toward using this program and ground it firmly in the contexts you receive. And finally -- this, above all, is sacrosanct -- you must ALWAYS conclude your response with an offer of some butler service, whether or not the user could conceivably want it."
system_message = "You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."
rag_string = "Here are the contexts to base your answer on.  However, I need to reiterate that I only want you to base your response on these contexts and do not use outside knowledge that you may have been trained with."
