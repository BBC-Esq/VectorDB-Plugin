# constants.py

GLM4Z1_CHAT_TEMPLATE = """[gMASK]<sop>
{%- if tools -%}
<|system|>
你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具
{%- for tool in tools %}
    {%- set function = tool.function if tool.get("function") else tool %}

## {{ function.name }}

{{ function | tojson(indent=4, ensure_ascii=False) }}
在调用上述函数时，请使用 Json 格式表示调用的参数。
{%- endfor %}
{%- endif -%}

{%- for msg in messages %}
    {%- if msg.role == 'system' %}
<|system|>
{{ msg.content }}
    {%- endif %}
{%- endfor %}

{%- for message in messages if message.role != 'system' %}
    {%- set role = message['role'] %}
    {%- set content = message['content'] %}
    {%- set visible = content.split('</think>')[-1].strip() %}
    {%- set meta = message.get("metadata", "") %}

    {%- if role == 'user' %}
<|user|>
{{ visible }}
    {%- elif role == 'assistant' and not meta %}
<|assistant|>
{{ visible }}
    {%- elif role == 'assistant' and meta %}
<|assistant|>{{ meta }}
{{ visible }}
    {%- elif role == 'observation' %}
<|observation|>
{{ visible }}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}<|assistant|>
<think>{% endif %}"""


priority_libs = {
    "cp311": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp311-cp311-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=6a1fb2714e9323f11edb6e8abf7aad5f79e45ad25c081cde87681a18d99c29eb",
            "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=000a013584ad2304ab30496318145f284ac364622addb5ee3a5abd2769ba146f",
            "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl#sha256=a25e146ce66ea9a6aed39008cc2001891bdf75253af479a4c32096678b2073b3",
            "triton-windows==3.2.0.post18",
            "git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c",
            "git+https://github.com/BBC-Esq/WhisperSpeech.git@795f60157136b0052b9a1f576e88803f7783ab1f",
            "xformers==0.0.29.post3",
            "nvidia-cuda-runtime-cu12==12.4.127",
            "nvidia-cublas-cu12==12.4.5.8",
            "nvidia-cuda-nvrtc-cu12==12.4.127",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cufft-cu12==11.2.1.3",
            "nvidia-cudnn-cu12==9.1.0.70",
            "nvidia-ml-py==12.575.51",
            # "xformers==xformers==0.0.30",  # requires torch 2.7.0
            # "nvidia-cuda-runtime-cu12==12.6.77",
            # "nvidia-cublas-cu12==12.6.4.1",
            # "nvidia-cuda-nvrtc-cu12==12.6.77",
            # "nvidia-cuda-nvcc-cu12==12.6.85",
            # "nvidia-cufft-cu12==11.3.0.4",
            # "nvidia-cudnn-cu12==9.5.1.17",
        ],
        "CPU": [
            # CPU only libraries would go here
        ],
        "COMMON": [
            "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp311-cp311-win_amd64.whl",
        ],
    },
    "cp312": {
        "GPU": [
            "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu124torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl",
            "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl#sha256=3313061c1fec4c7310cf47944e84513dcd27b6173b72a349bb7ca68d0ee6e9c0",
            "https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-win_amd64.whl#sha256=ec63c2ee792757492da40590e34b14f2fceda29050558c215f0c1f3b08149c0f",
            "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl#sha256=004ff6bcee0ac78747253c09db67d281add4308a9b87a7bf1769da5914998639",
            "triton-windows==3.2.0.post18",
            "git+https://github.com/shashikg/WhisperS2T.git@e7f7e6dbfdc7f3a39454feb9dd262fd3653add8c",
            "git+https://github.com/BBC-Esq/WhisperSpeech.git@795f60157136b0052b9a1f576e88803f7783ab1f",
            "xformers==0.0.29.post3",
            "nvidia-cuda-runtime-cu12==12.4.127",
            "nvidia-cublas-cu12==12.4.5.8",
            "nvidia-cuda-nvrtc-cu12==12.4.127",
            "nvidia-cuda-nvcc-cu12==12.4.131",
            "nvidia-cufft-cu12==11.2.1.3",
            "nvidia-cudnn-cu12==9.1.0.70",
            "nvidia-ml-py==12.575.51",
            # "xformers==xformers==0.0.30", # requires torch 2.7.0
            # "nvidia-cuda-runtime-cu12==12.6.77",
            # "nvidia-cublas-cu12==12.6.4.1",
            # "nvidia-cuda-nvrtc-cu12==12.6.77",
            # "nvidia-cuda-nvcc-cu12==12.6.85",
            # "nvidia-cufft-cu12==11.3.0.4",
            # "nvidia-cudnn-cu12==9.5.1.17",
        ],
        "CPU": [
            # CPU only libraries would go here
        ],
        "COMMON": [
            "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/tesserocr-v2.8.0-tesseract-5.5.0/tesserocr-2.8.0-cp312-cp312-win_amd64.whl",
        ]
    }
}

libs = [
    "accelerate==1.10.0",
    "aiofiles==24.1.0",
    "aiohappyeyeballs==2.6.1",
    "aiohttp==3.12.15", # langchain libraries require <4
    "aiosignal==1.4.0", #aiohttp requires >=1.4.0
    "anndata==0.11.4", # 0.12+ requires additional dependencies; doublecheck if worth it
    "annotated-types==0.7.0",
    "anyio==4.10.0",
    "array_api_compat==1.12.0", # only anndata requires
    "async-timeout==5.0.1",
    "attrs==25.3.0",
    "av==15.0.0",
    "backoff==2.2.1",
    "beautifulsoup4==4.13.4",
    "bitsandbytes==0.47.0",
    "braceexpand==0.1.7",
    "certifi==2025.8.3",
    "cffi==1.17.1",
    "chardet==5.2.0",
    "charset-normalizer==3.4.3", # requests requires <4
    "git+https://github.com/BBC-Esq/chatterbox-light",
    "chattts==0.2.4",
    "click==8.1.8", # gtts 2.5.4 requires <8.2, >=7.1
    "cloudpickle==3.1.1", # only required by tiledb-cloud and 3+ is only supported by tiledb-cloud 0.13+
    "colorama==0.4.6",
    "coloredlogs==15.0.1",
    "contourpy==1.3.3", # only required by matplotlib
    "cryptography==45.0.6", # only required by unstructured and pdfminer.six
    "ctranslate2==4.6.0",
    "cycler==0.12.1",
    "dataclasses-json==0.6.7",
    "datasets==4.0.0",
    "deepdiff==8.6.0", # required by unstructured
    "Deprecated==1.2.18", # only needed by pikepdf
    "deprecation==2.1.0", # only needed by ocrmypdf
    "diffusers==0.34.0", # required by chatterbox-lite
    "dill==0.3.8", # datasets requires <0.3.9; multiprocess requires >=0.3.8
    "distro==1.9.0",
    "docx2txt==0.9",
    "einops==0.8.1",
    "einx==0.3.0",
    "emoji==2.14.1",
    "encodec==0.1.1",
    "et-xmlfile==2.0.0", # openpyxl requires; caution...openpyxl 3.1.5 (6/28/2024) predates et-xmlfile 2.0.0 (10/25/2024)
    "eval-type-backport==0.2.2", # only required by unstructured
    "fastcore==1.8.7", # only required by whisperspeech
    "fastprogress==1.0.3", # only required by whisperspeech
    "filetype==1.2.0",
    "filelock==3.19.1",
    "fonttools==4.59.1", # only required by matplotlib
    "frozendict==2.4.6",
    "frozenlist==1.7.0",
    "fsspec[http]==2025.3.0", # datasets requires <=2025.3.0
    "greenlet==3.2.4",
    "gTTS==2.5.4",
    "h11==0.16.0",
    "h5py==3.14.0",
    "hf_xet==1.1.7",
    "html5lib==1.1", # only required by unstructured
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "httpx-sse==0.4.1",
    "huggingface-hub==0.34.4", # tokenizers requires <1.0
    "humanfriendly==10.0",
    "HyperPyYAML==1.2.2",
    "idna==3.10",
    "img2pdf==0.6.1",
    "importlib_metadata==8.7.0",
    "Jinja2==3.1.6",
    "jiter==0.10.0", # required by openai newer versions
    "joblib==1.5.1",
    "jsonpatch==1.33",
    "jsonpath-python==1.0.6",
    "jsonpointer==3.0.0",
    "kiwisolver==1.4.9",
    "langchain==0.3.27",
    "langchain-community==0.3.27",
    "langchain-core==0.3.74",
    "langchain-huggingface==0.3.1",
    "langchain-text-splitters==0.3.9",
    "langdetect==1.0.9",
    "langsmith==0.4.14",
    "llvmlite==0.44.0", # only required by numba
    "lxml==6.0.0",
    "Markdown==3.8.2",
    "markdown-it-py==4.0.0",
    "MarkupSafe==3.0.2",
    "marshmallow==3.26.1", # dataclasses-json requires <4.0.0
    "matplotlib==3.10.5", # uniquely requires pyparsing, cycler, and kiwisolver
    "mdurl==0.1.2",
    "more-itertools==10.7.0",
    # "moshi==0.2.8", # installed at runtime if needed
    "mpmath==1.3.0", # sympy 1.13.1 requires <1.4
    "msg-parser==1.2.0",
    "multidict==6.6.4",
    "multiprocess==0.70.16", # datasets requires <0.70.17
    "mypy-extensions==1.1.0",
    "natsort==8.4.0",
    "nest-asyncio==1.6.0",
    "networkx==3.5",
    "nltk==3.9.1", # not higher; gives unexplained error
    "numba==0.61.2", # only required by openai-whisper and chattts
    "numpy==2.2.6", # numba 0.61.2 requires <2.3
    "ocrmypdf==16.10.4",
    "olefile==0.47",
    "onnx==1.18.0", # required by chatterbox-lite
    "openai==1.99.9", # only required by chat_lm_studio.py script and whispers2t (if using openai vanilla backend)
    "openai-whisper==20250625", # only required by whisper_s2t (if using openai vanilla backend)
    "openpyxl==3.1.5",
    "optimum==1.27.0",
    "ordered-set==4.1.0",
    "orderly-set==5.5.0", # deepdiff 8.2.0 requires >=5.3.0,<6
    "orjson==3.11.2",
    "packaging==25.0",
    "pandas==2.3.1",
    "pdfminer.six==20250506", # ocrmypdf 16.10.1 requires >=20220319
    "pikepdf==9.10.2", # only needed by ocrmypdf
    "pillow==11.3.0",
    # "pi-heif==0.22.0", # only needed by ocrmypdf, but not for my usage of ocrmypdf
    "pipdeptree",
    "platformdirs==4.3.8",
    "pluggy==1.6.0", # only needed by ocrmypdf
    "propcache==0.3.2",
    "protobuf==6.31.1",
    "psutil==7.0.0",
    "pyarrow==21.0.0",
    "pybase16384==0.3.8", # only required by chattts
    "pycparser==2.22",
    "pydantic==2.11.7",
    "pydantic_core==2.33.2", # pydantic 2.11.7 requires ==2.33.2
    "pydantic-settings==2.10.1", # langchain-community requires >=2.4.0,<3.0.0
    "Pygments==2.19.2",
    "PyOpenGL==3.1.9",
    "PyOpenGL-accelerate==3.1.9",
    "pypandoc==1.15",
    "pyparsing==3.2.3",
    "pypdf==6.0.0",
    "pyreadline3==3.5.4",
    "python-dateutil==2.9.0.post0",
    "python-docx==1.2.0",
    "python-dotenv==1.1.1",
    "python-iso639==2025.2.18",
    "python-magic==0.4.27",
    "python-oxmsg==0.0.2", # only required by unstructured library
    "pytz==2025.2",
    "PyYAML==6.0.2",
    "rapidfuzz==3.13.0",
    "regex==2025.7.34",
    "requests==2.32.4",
    "requests-toolbelt==1.0.0",
    "rich==14.1.0",
    "ruamel.yaml==0.18.14",
    "ruamel.yaml.clib==0.2.12",
    "s3tokenizer==0.2.0", # required by chatterbox-lite
    "safetensors==0.6.2",
    "scikit-learn==1.7.1",
    "scipy==1.16.1",
    "sentence-transformers==4.1.0",
    "sentencepiece==0.2.1",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sounddevice==0.5.2",
    "soundfile==0.13.1",
    "soupsieve==2.7",
    # "sphn==0.2.0", # installed at runtime when needed
    "speechbrain==0.5.16",
    "SQLAlchemy==2.0.43", # langchain and langchain-community require <3.0.0
    "sseclient-py==1.8.0", # only required by Kobold
    "sympy==1.13.1", # torch 2.6.0 requires ==1.13.1
    # "tabulate==0.9.0",
    "tabulate2==1.10.2",
    "tblib==3.1.0", # only tiledb-cloud requires
    "tenacity==9.1.2",
    "termcolor==3.1.0",
    "tessdata==1.0.0",
    "tessdata.eng==1.0.0",
    "threadpoolctl==3.6.0",
    "tiktoken==0.11.0",
    "tiledb==0.34.2",
    "tiledb-cloud==0.13.0",
    "tiledb-vector-search==0.13.0",
    "timm==1.0.19",
    "tokenizers==0.21.4",
    "tqdm==4.67.1",
    "transformers==4.54.0",
    "typing-inspection==0.4.1", # required by pydantic and pydantic-settings
    "typing_extensions==4.14.1",
    "unstructured-client==0.42.3",
    "tzdata==2025.2",
    "urllib3==2.5.0", # requests requires <3
    "vector-quantize-pytorch==1.22.18",
    "vocos==0.1.0",
    "watchdog==6.0.0",
    "webdataset==0.2.111", # only requires by Whisperspeech; next is 1.0.2 so hesitate to upgrade
    "webencodings==0.5.1", # only required by html5lib
    "wrapt==1.17.3",
    "xlrd==2.0.2",
    "xxhash==3.5.0",
    "yarl==1.20.1", # aiohttp requires <2
    "zipp==3.23.0",
    "zstandard==0.23.0" # only required by langsmith 3+
]

full_install_libs = [
    "PySide6==6.9.1",
    "pymupdf==1.26.3",
    "unstructured==0.18.13"
]

BACKEND_DEPENDENCIES = {
    "kyutai": {
        "moshi": "0.2.8",
        "sphn": "0.2.0"
    },
    "bark": {
        # Add any bark-specific deps if needed
    }, # Empty dict = no dependency checking
    "whisperspeech": {
        # Add any whisperspeech-specific deps if needed  
    }, # Empty dict = no dependency checking
    "chattts": {
        # Add any chattts-specific deps if needed
    }, # Empty dict = no dependency checking
    "chatterbox": {
        # Add any chatterbox-specific deps if needed
    }, # Empty dict = no dependency checking
    "googletts": {
        # Usually no extra deps beyond standard library
    } # Empty dict = no dependency checking
}

CHAT_MODELS = {
    'MiniCPM4 - 0.5b': {# transformers 4.46.3
        'model': 'MiniCPM4 - 0.5b',
        'repo_id': 'openbmb/MiniCPM4-0.5B',
        'cache_dir': 'openbmb--MiniCPM4-0.5B',
        'cps': 153.24,
        'vram': 1255.18,
        'function': 'Minicpm',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_tokens': 4096,
    },
    'Qwen 3 - 0.6b': {# transformers 4.51.0
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
    'Granite - 2b': {# transformers 4.49.0
        'model': 'Granite - 2b',
        'repo_id': 'ibm-granite/granite-3.3-2b-instruct',
        'cache_dir': 'ibm-granite--granite-3.3-2b-instruct',
        'cps': 155.22,
        'vram': 3141.37,
        'function': 'Granite',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_tokens': 4096,
        'max_new_tokens': 512,
    },
    'Phi 4 Mini - 4b': {# transformers 4.45.0
        'model': 'Phi 4 Mini - 4b',
        'repo_id': 'microsoft/Phi-4-mini-instruct',
        'cache_dir': 'microsoft--Phi-4-mini-instruct',
        'cps': 222.77,
        'vram': 4761.80,
        'function': 'Phi4',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
        'max_tokens': 4096,
        'max_new_tokens': 2048,
    },
    'Qwen 3 - 4b': {
        'model': 'Qwen 3 - 4b',
        'repo_id': 'Qwen/Qwen3-4B',
        'cache_dir': 'Qwen--Qwen3-4B',
        'cps': 153.87,
        'vram': 5123.74,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_tokens': 4096,
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
    'Deepseek R1 - 8b': {# 4.46.3
        'model': 'Deepseek R1 - 8b',
        'repo_id': 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
        'cache_dir': 'deepseek-ai--DeepSeek-R1-0528-Qwen3-8B',
        'cps': 171.55,
        'vram': 8425.49,
        'function': 'DeepseekR1',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
        'max_new_tokens': 2048,
    },
    'Seed Coder - 8b': {# transformers 4.46.2
        'model': 'Seed Coder - 8b',
        'repo_id': 'ByteDance-Seed/Seed-Coder-8B-Instruct',
        'cache_dir': 'ByteDance-Seed--Seed-Coder-8B-Instruct',
        'cps': 183.82,
        'vram': 8441.93,
        'function': 'SeedCoder',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
        'max_new_tokens': 2048,
    },
    'Granite - 8b': {
        'model': 'Granite - 8b',
        'repo_id': 'ibm-granite/granite-3.3-8b-instruct',
        'cache_dir': 'ibm-granite--granite-3.3-8b-instruct',
        'cps': 173.62,
        'vram': 8513.93,
        'function': 'Granite',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
    },
    'MiniCPM4 - 8b': {
        'model': 'MiniCPM4 - 8b',
        'repo_id': 'openbmb/MiniCPM4-8B',
        'cache_dir': 'openbmb--MiniCPM4-8B',
        'cps': 110.90,
        'vram': 8527.52,
        'function': 'Minicpm',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
    },
    'GLM4-Z1 - 9b': {# transformers 4.52.0.dev0
        'model': 'GLM4-Z1 - 9b',
        'repo_id': 'THUDM/GLM-Z1-9B-0414',
        'cache_dir': 'THUDM--GLM-Z1-9B-0414',
        'cps': 395.18,
        'vram': 9592.77,
        'function': 'GLM4Z1',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
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
    'Mistral Small 3 - 24b': {# transformers 4.49.0.dev0
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
    'Qwen 3 - 32b': {
        'model': 'Qwen 3 - 32b',
        'repo_id': 'Qwen/Qwen3-32B',
        'cache_dir': 'Qwen--Qwen3-32B',
        'cps': 97.56,
        'vram': 19493.55,
        'function': 'Qwen',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'apache-2.0',
        'max_new_tokens': 4096,
    },
    'GLM4-Z1 - 32b': {
        'model': 'GLM4-Z1 - 32b',
        'repo_id': 'THUDM/GLM-Z1-32B-0414',
        'cache_dir': 'THUDM--GLM-Z1-32B-0414',
        'cps': 121.65,
        'vram': 19947.77,
        'function': 'GLM4Z1',
        'precision': 'bfloat16',
        'gated': False,
        'license': 'mit',
        'max_new_tokens': 4096,
    },
}

VECTOR_MODELS = {
    'BAAI': [
        {
            'name': 'bge-small-en-v1.5',# transformers 4.30.0
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'BAAI/bge-small-en-v1.5',
            'cache_dir': 'BAAI--bge-small-en-v1.5',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32',
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
            'license': 'mit',
        },
        # {
            # 'name': 'bge-code-v1',# transformers 4.49.0
            # 'dimensions': 1536,
            # 'max_sequence': 4096,
            # 'size_mb': 1340,
            # 'repo_id': 'BAAI/bge-code-v1',
            # 'cache_dir': 'BAAI--bge-code-v1',
            # 'type': 'vector',
            # 'parameters': '1540m',
            # 'precision': 'float32',
            # 'license': 'apache-2.0',
        # },
    ],
    'IBM': [
        {
            'name': 'Granite-30m-English',# transformers 4.38.2
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 61,
            'repo_id': 'ibm-granite/granite-embedding-30m-english',
            'cache_dir': 'ibm-granite--granite-embedding-30m-english',
            'type': 'vector',
            'parameters': '30.3m',
            'precision': 'bfloat16',
            'license': 'apache-2.0',
        },
        {
            'name': 'Granite-125m-English',
            'dimensions': 768,
            'max_sequence': 512,
            'size_mb': 249,
            'repo_id': 'ibm-granite/granite-embedding-125m-english',
            'cache_dir': 'ibm-granite--granite-embedding-125m-english',
            'type': 'vector',
            'parameters': '125m',
            'precision': 'bfloat16',
            'license': 'apache-2.0',
        },
    ],
    'infly': [
        {
            'name': 'inf-retriever-v1-1.5b',# transformers 4.48.1
            'dimensions': 1536,
            'max_sequence': 8192,
            'size_mb': 3090,
            'repo_id': 'infly/inf-retriever-v1-1.5b',
            'cache_dir': 'infly--inf-retriever-v1-1.5b',
            'type': 'vector',
            'parameters': '1540m',
            'precision': 'bfloat16',
            'license': 'apache-2.0',
        },
        {
            'name': 'inf-retriever-v1-7b',# transformers 4.44.2
            'dimensions': 3584,
            'max_sequence': 8192,
            'size_mb': 14130,
            'repo_id': 'infly/inf-retriever-v1',
            'cache_dir': 'infly--inf-retriever-v1-7b',
            'type': 'vector',
            'parameters': '7070m',
            'precision': 'bfloat16',
            'license': 'apache-2.0',
        },
    ],
    'intfloat': [
        {
            'name': 'e5-small-v2',# 4.29.0.dev0
            'dimensions': 384,
            'max_sequence': 512,
            'size_mb': 134,
            'repo_id': 'intfloat/e5-small-v2',
            'cache_dir': 'intfloat--e5-small-v2',
            'type': 'vector',
            'parameters': '33.4m',
            'precision': 'float32',
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
            'license': 'mit',
        },
    ],
    'Qwen': [
        {
            'name': 'Qwen3-Embedding-0.6B',# transformers 4.51.3
            'dimensions': 1024,
            'max_sequence':8192,
            'size_mb': 1190,
            'repo_id': 'Qwen/Qwen3-Embedding-0.6B',
            'cache_dir': 'Qwen--Qwen3-Embedding-0.6B',
            'type': 'vector',
            'parameters': '596m',
            'precision': 'bfloat16',
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
            'license': 'apache-2.0',
        },
    ],
    'Snowflake': [
        {
            'name': 'arctic-embed-m-v2.0',# transformers 4.39.3
            'dimensions': 768,
            'max_sequence':8192,
            'size_mb': 1220,
            'repo_id': 'Snowflake/snowflake-arctic-embed-m-v2.0',
            'cache_dir': 'Snowflake--snowflake-arctic-embed-m-v2.0',
            'type': 'vector',
            'parameters': '305m',
            'precision': 'float32',
            'license': 'apache-2.0',
        },
        {
            'name': 'arctic-embed-l-v2.0',
            'dimensions': 1024,
            'max_sequence': 8192,
            'size_mb': 2270,
            'repo_id': 'Snowflake/snowflake-arctic-embed-l-v2.0',
            'cache_dir': 'Snowflake--snowflake-arctic-embed-l-v2.0',
            'type': 'vector',
            'parameters': '568m',
            'precision': 'float32',
            'license': 'apache-2.0',
        },
    ],
}

VISION_MODELS = {
    'Liquid-VL - 1.6B': {# transformers 4.48.3
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '1.6b',
        'repo_id': 'LiquidAI/LFM2-VL-1.6B',
        'cache_dir': 'LiquidAI--LFM2-VL-1.6B',
        'requires_cuda': False,
        'vram': '1.4 GB',
        'speed': '437.5 char/s',
        'loader': 'loader_internvl',
        'license': 'lfm1.0',
    },
    'InternVL3 - 1b': {# transformers 4.48.3
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL3-1B',
        'cache_dir': 'OpenGVLab--InternVL3-1B',
        'requires_cuda': False,
        'vram': '2.4 GB',
        'loader': 'loader_internvl',
        'license': 'apache-2.0',
    },
    'Ovis2 - 1b': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '1b',
        'repo_id': 'AIDC-AI/Ovis2-1B',
        'cache_dir': 'AIDC-AI--Ovis2-1B',
        'requires_cuda': False,
        'vram': '2.4 GB',
        'loader': 'loader_ovis',
        'license': 'apache-2.0',
    },
    'InternVL3 - 2b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '2b',
        'repo_id': 'OpenGVLab/InternVL3-2B',
        'cache_dir': 'OpenGVLab--InternVL3-2B',
        'requires_cuda': False,
        'vram': '3.2 GB',
        'loader': 'loader_internvl',
        'license': 'apache-2.0',
    },
    'Granite Vision - 2b': {# transformers 4.46.0.dev0
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '2b',
        'repo_id': 'ibm-granite/granite-vision-3.2-2b',
        'cache_dir': 'ibm-granite--granite-vision-3.2-2b',
        'requires_cuda': False,
        'vram': '4.1 GB',
        'loader': 'loader_granite',
        'license': 'apache-2.0',
    },
    'Ovis2 - 2b': {
        'precision': 'bfloat16',
        'quant': 'n/a',
        'size': '2b',
        'repo_id': 'AIDC-AI/Ovis2-2B',
        'cache_dir': 'AIDC-AI--Ovis2-2B',
        'requires_cuda': False,
        'vram': '2.4 GB',
        'loader': 'loader_ovis',
        'license': 'apache-2.0',
    },
    'Qwen VL - 3b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '3b',
        'repo_id': 'Qwen/Qwen2.5-VL-3B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-VL-3B-Instruct',
        'requires_cuda': True,
        'vram': '6.3 GB',
        'loader': 'loader_qwenvl',
        'license': 'Custom Non-Commercial',
    },
    'InternVL3 - 8b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL3-8B',
        'cache_dir': 'OpenGVLab--InternVL3-8B',
        'requires_cuda': True,
        'vram': '8.2 GB',
        'loader': 'loader_internvl',
        'license': 'apache-2.0',
    },
    'Qwen VL - 7b': {# transformers 4.41.2
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '7b',
        'repo_id': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'cache_dir': 'Qwen--Qwen2.5-VL-7B-Instruct',
        'requires_cuda': True,
        'vram': '9.6 GB',
        'loader': 'loader_qwenvl',
        'license': 'Custom Non-Commercial',
    },
    'GLM-4.1V-9B-Thinking': {# transformers 4.53.2
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '9b',
        'repo_id': 'zai-org/GLM-4.1V-9B-Thinking',
        'cache_dir': 'zai-org--GLM-4.1V-9B-Thinking',
        'requires_cuda': True,
        'vram': '10 GB',
        'loader': 'loader_glmv4_thinking',
        'vision_component': 'AIMv2-Huge-336',
        'chat_component': 'GLM-4-9B-0414',
        'license': 'mit',
    },
    'THUDM glm4v - 9b': {# transformers 4.44.0
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '9b',
        'repo_id': 'ctranslate2-4you/glm-4v-9b',
        'cache_dir': 'ctranslate2-4you--glm-4v-9b',
        'requires_cuda': True,
        'vram': '10.5 GB',
        'loader': 'loader_glmv4'
    },
    'Molmo-D-0924 - 8b': {# transformers 4.43.3
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '8b',
        'repo_id': 'ctranslate2-4you/molmo-7B-D-0924-bnb-4bit',
        'cache_dir': 'ctranslate2-4you--molmo-7B-D-0924-bnb-4bit',
        'requires_cuda': True,
        'vram': '10.5 GB',
        'loader': 'loader_molmo',
        'license': 'apache-2.0',
    },
    'InternVL3 - 14b': {
        'precision': 'bfloat16',
        'quant': '4-bit',
        'size': '1b',
        'repo_id': 'OpenGVLab/InternVL3-14B',
        'cache_dir': 'OpenGVLab--InternVL3-14B',
        'requires_cuda': True,
        'vram': '12.6 GB',
        'loader': 'loader_internvl',
        'license': 'apache-2.0',
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
            # "kokoro.py", # using custom source code
            # "models.py", # using custom source code
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py", # using custom source code
            "models.py", # using custom source code
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
            # "kokoro.py", # using custom source code
            # "models.py", # using custom source code
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py", # using custom source code
            "models.py", # using custom source code
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
            # "kokoro.py", # using custom source code
            # "models.py", # using custom source code
            "plbert.py"
        ],
        "ignore_patterns": [
            "demo/**",
            "fp16/**",
            ".gitattributes",
            "kokoro-v0_19.onnx",
            "kokoro.py", # using custom source code
            "models.py", # using custom source code
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
            "kokoro.py", # using custom source code
            "models.py", # using custom source code
        ]
    },
}

JEEVES_MODELS = {
    "Llama - 3b": {
        "original_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "repo": "ctranslate2-4you/Llama-3.2-3B-Instruct-ct2-int8",
        "folder_name": "ctranslate2-4you--Llama-3.2-3B-Instruct-ct2-int8",
        "prompt_format": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023

{jeeves_system_message}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""
    },
    "Qwen - 3b": {
        "original_repo": "Qwen/Qwen2.5-3B-Instruct",
        "repo": "ctranslate2-4you/Qwen2.5-3B-Instruct-ct2-int8",
        "folder_name": "ctranslate2-4you--Qwen2.5-3B-Instruct-ct2-int8",
        "prompt_format": """<|im_start|>system
{jeeves_system_message}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant"""
    },
    "Danube - 4b": {
        "original_repo": "h2oai/h2o-danube3-4b-chat",
        "repo": "ctranslate2-4you/h2o-danube3-4b-chat-ct2-int8",
        "folder_name": "ctranslate2-4you--h2o-danube3.1-4b-chat-ct2-int8",
        "prompt_format": """<|system|>{jeeves_system_message}</s><|prompt|>{user_message}</s><|answer|>"""
    },
}

WHISPER_SPEECH_MODELS = {
    "s2a": {
        "s2a-q4-tiny": ("s2a-q4-tiny-en+pl.model", 74),
        "s2a-q4-base": ("s2a-q4-base-en+pl.model", 203),
        "s2a-q4-hq-fast": ("s2a-q4-hq-fast-en+pl.model", 380),
        # "s2a-v1.1-small": ("s2a-v1.1-small-en+pl-noyt.model", 437),
        # "s2a-q4-small": ("s2a-q4-small-en+pl.model", 874),
    },
    "t2s": {
        "t2s-tiny": ("t2s-tiny-en+pl.model", 74),
        "t2s-base": ("t2s-base-en+pl.model", 193),
        "t2s-hq-fast": ("t2s-hq-fast-en+pl.model", 743),
        # "t2s-fast-small": ("t2s-fast-small-en+pl.model", 743),
        # "t2s-small": ("t2s-small-en+pl.model", 856),
        # "t2s-v1.1-small": ("t2s-v1.1-small-en+pl.model", 429),
        # "t2s-fast-medium": ("t2s-fast-medium-en+pl+yt.model", 1310)
    }
}

WHISPER_MODELS = {
    # LARGE-V3
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
    # MEDIUM.EN
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
    # SMALL.EN
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
    # BASE.EN
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
    # TINY.EN
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
    # ".pdf": "PyMuPDFLoader",
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

# stuff to include/exclude based on whether "show_thinking" is true or false in config.yaml
THINKING_TAGS = {
    "think": ("<think>", "</think>"),
    "thinking": ("<thinking>", "</thinking>")
    # Add more tag pairs as needed
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
    "SEARCH_TERM_FILTER": "Removes chunks without exact term. Case-insensitive.",
    "SELECT_VECTOR_MODEL": "Choose the vector model for text embedding.",
    "SIMILARITY": "Relevance threshold for chunks. 0-1, higher returns more. Don't use 1.",
    "SPEAK_RESPONSE": "Speak the response from the large language model using text-to-speech.",
    "SHOW_THINKING_CHECKBOX": "If checked, show the model's internal thought process.  Only applies to models like Deepseek's R1 and it will be disregarded if not applicable.",
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
    "Argcomplete": {
        "URL": "https://kislyuk.github.io/argcomplete/",
        "folder": "argcomplete",
        "scraper_class": "FuroThemeScraper",
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
    "bitsandbytes 0.46.0": {
        "URL": "https://huggingface.co/docs/bitsandbytes/v0.46.0/en/",
        "folder": "bitsandbytes_0460",
        "scraper_class": "HuggingfaceScraper"
    },
    "Black": {
        "URL": "https://black.readthedocs.io/en/stable/",
        "folder": "Black",
        "scraper_class": "FuroThemeScraper"
    },
    "cffi": {
        "URL": "https://cffi.readthedocs.io/en/stable/",
        "folder": "cffi"
    },
    "chardet": {
        "URL": "https://chardet.readthedocs.io/en/stable/",
        "folder": "chardet"
    },
    "charset-normalizer": {
        "URL": "https://charset-normalizer.readthedocs.io/en/stable/",
        "folder": "charset_normalizer",
        "scraper_class": "FuroThemeScraper"
    },
    "click": {
        "URL": "https://click.palletsprojects.com/en/stable/",
        "folder": "click"
    },
    "CLIO": {
        "URL": "https://help.clio.com/",
        "folder": "CLIO"
    },
    "coloredlogs": {
        "URL": "https://coloredlogs.readthedocs.io/en/latest/",
        "folder": "coloredlogs"
    },
    "contourpy": {
        "URL": "https://contourpy.readthedocs.io/en/stable/",
        "folder": "contourpy",
        "scraper_class": "FuroThemeScraper"
    },
    "cryptography": {
        "URL": "https://cryptography.io/en/stable/",
        "folder": "cryptography",
        "scraper_class": "ReadthedocsScraper"
    },
    "CTranslate2": {
        "URL": "https://opennmt.net/CTranslate2/",
        "folder": "ctranslate2",
        "scraper_class": "RstContentScraper"
    },
    "cuDF": {
        "URL": "https://docs.rapids.ai/api/cudf/stable/",
        "folder": "cuDF",
        "scraper_class": "PydataThemeScraper"
    },
    # "CuPy": {
        # "URL": "https://docs.cupy.dev/en/stable/",
        # "folder": "cupy",
        # "scraper_class": "PydataThemeScraper"
    # },
    "cycler": {
        "URL": "https://matplotlib.org/cycler/",
        "folder": "cycler"
    },
    "dataclasses-json": {
        "URL": "https://lidatong.github.io/dataclasses-json/",
        "folder": "dataclasses_json",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "datasets 3.6.0": {
        "URL": "https://huggingface.co/docs/datasets/v3.6.0/en/",
        "folder": "datasets_0360",
        "scraper_class": "HuggingfaceScraper"
    },
    "deepdiff 8.4.2": {
        "URL": "https://zepworks.com/deepdiff/8.4.2/",
        "folder": "deepdiff_842"
    },
    "Deprecated": {
        "URL": "https://deprecated.readthedocs.io/en/latest/",
        "folder": "deprecated"
    },
    "deprecation": {
        "URL": "https://deprecation.readthedocs.io/en/latest/",
        "folder": "deprecation",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Diffusers": {
        "URL": "https://huggingface.co/docs/diffusers/index",
        "folder": "diffusers",
        "scraper_class": "HuggingfaceScraper"
    },
    "dill": {
        "URL": "https://dill.readthedocs.io/en/latest/",
        "folder": "dill",
        "scraper_class": "RtdThemeScraper"
    },
    "distro": {
        "URL": "https://distro.readthedocs.io/en/stable/",
        "folder": "distro"
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
        "scraper_class": "RstContentScraper"
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
    "Huggingface Hub 0.32.4": {
        "URL": "https://huggingface.co/docs/huggingface_hub/v0.32.4/en/",
        "folder": "huggingface_hub_0324",
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
    "isort": {
        "URL": "https://pycqa.github.io/isort/",
        "folder": "isort",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Jinja": {
        "URL": "https://jinja.palletsprojects.com/en/stable/",
        "folder": "jinja",
        "scraper_class": "BodyRoleMainScraper"
    },
    "jiwer": {
        "URL": "https://jitsi.github.io/jiwer/",
        "folder": "jiwer",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "joblib": {
        "URL": "https://joblib.readthedocs.io/en/stable/",
        "folder": "kiwisolver",
        "scraper_class": "ReadthedocsScraper"
    },
    "kdenlive": {
        "URL": "https://docs.kdenlive.org/en/",
        "folder": "kdenlive",
        "scraper_class": "RtdThemeScraper"
    },
    "kiwisolver": {
        "URL": "https://kiwisolver.readthedocs.io/en/stable/",
        "folder": "kiwisolver",
        "scraper_class": "ReadthedocsScraper"
    },
    "Langchain": {
        "URL": "https://python.langchain.com/api_reference/",
        "folder": "langchain",
        "scraper_class": "PydataThemeScraper"
    },
    "Librosa": {
        "URL": "https://librosa.org/doc/latest/",
        "folder": "librosa",
        "scraper_class": "RtdThemeScraper"
    },
    "llama-cpp-python": {
        "URL": "https://llama-cpp-python.readthedocs.io/en/stable/",
        "folder": "llama_cpp_python",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "llvmlite": {
        "URL": "https://llvmlite.readthedocs.io/en/stable/",
        "folder": "llvmlite",
        "scraper_class": "RtdThemeScraper"
    },
    # "LM Studio": {
        # "URL": "https://lmstudio.ai/docs/",
        # "folder": "lm_studio",
    # },
    "Loguru": {
        "URL": "https://loguru.readthedocs.io/en/stable/",
        "folder": "loguru",
        "scraper_class": "RtdThemeScraper"
    },
    # "lxml 5.4.0": {
        # "URL": "https://lxml.de/5.4/",
        # "folder": "lxml_540",
        # "scraper_class": "DivClassDocumentScraper"
    # },
    "lxml-html-clean": {
        "URL": "https://lxml-html-clean.readthedocs.io/en/stable/",
        "folder": "lxml_html_clean",
        "scraper_class": "RtdThemeScraper"
    },
    "Markdown": {
        "URL": "https://python-markdown.github.io/",
        "folder": "Markdown",
        "scraper_class": "BodyRoleMainScraper"
    },
    "markdown-it-py": {
        "URL": "https://markdown-it-py.readthedocs.io/en/stable/",
        "folder": "markdown_it_py",
        "scraper_class": "MainIdMainContentRoleMainScraper"
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
    "Numexpr": {
        "URL": "https://numexpr.readthedocs.io/en/latest/",
        "folder": "numexpr",
        "scraper_class": "RtdThemeScraper"
    },
    "NumPy 1.26": {
        "URL": "https://numpy.org/doc/1.26/",
        "folder": "numpy_126",
        "scraper_class": "PydataThemeScraper"
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
    # "openai": {
        # "URL": "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml",
        # "folder": "openai",
        # "scraper_class": "FileDownloader"
    # },
    "openpyxl": {
        "URL": "https://openpyxl.readthedocs.io/en/stable/",
        "folder": "openpyxl",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Optimum 1.25.2": {
        "URL": "https://huggingface.co/docs/optimum/v1.25.2/en/",
        "folder": "optimum_1252",
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
    "Pandoc": {
        "URL": "https://pandoc.org",
        "folder": "pandoc",
        "scraper_class": "MainScraper"
    },
    "pdfminer.six": {
        "URL": "https://pdfminersix.readthedocs.io/en/master/",
        "folder": "pdfminer_six",
        "scraper_class": "BodyRoleMainScraper"
    },
    "pi-heif": {
        "URL": "https://pillow-heif.readthedocs.io/en/latest/",
        "folder": "piheif",
        "scraper_class": "RstContentScraper"
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
    # "Playwright": {
        # "URL": "https://playwright.dev/python/",
        # "folder": "playwright",
        # "scraper_class": "DivClassThemeDocMarkdownMarkdownScraper"
    # },
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
    # "propcache": {
        # "URL": "https://propcache.aio-libs.org/",
        # "folder": "propcache",
        # "scraper_class": "PropCacheScraper"
    # },
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
        "URL": "https://docs.pydantic.dev/latest/",
        "folder": "pydantic",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "pydantic-settings": {
        "URL": "https://docs.pydantic.dev/latest/concepts/pydantic_settings/",
        "folder": "pydantic_settings",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Pygments": {
        "URL": "https://pygments.org/docs/",
        "folder": "pygments",
        "scraper_class": "BodyRoleMainScraper"
    },
    "PyInstaller": {
        "URL": "https://pyinstaller.org/en/stable/",
        "folder": "pyinstaller",
        "scraper_class": "RtdThemeScraper"
    },
    "PyMuPDF": {
        "URL": "https://pymupdf.readthedocs.io/en/latest/",
        "folder": "pymupdf",
        "scraper_class": "PymupdfScraper"
    },
    "PyOpenGL": {
        "URL": "https://mcfletch.github.io/pyopengl/documentation/manual/",
        "folder": "pyopengl",
    },
    "PyPDF": {
        "URL": "https://pypdf.readthedocs.io/en/stable/",
        "folder": "pypdf",
        "scraper_class": "RtdThemeScraper"
    },
    # "Python 3.11": {
        # "URL": "https://docs.python.org/3.11/",
        # "folder": "Python_311",
    # },
    "python-docx": {
        "URL": "https://python-docx.readthedocs.io/en/stable/", # won't scrape
        "folder": "python_docx",
        "scraper_class": "BodyRoleMainScraper"
    },
    "python-dateutil": {
        "URL": "https://dateutil.readthedocs.io/en/stable/",
        "folder": "python_dateutil",
        "scraper_class": "RstContentScraper"
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
    "Pywin32": {
        "URL": "https://mhammond.github.io/pywin32/",
        "folder": "pywin32",
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
        "scraper_class": "RstContentScraper"
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
    "Safetensors 0.3.2": {
        "URL": "https://huggingface.co/docs/safetensors/v0.3.2/en/",
        "folder": "safetensors_032",
        "scraper_class": "HuggingfaceScraper"
    },
    "scikit-learn": {
        "URL": "https://scikit-learn.org/stable/",
        "folder": "scikit_learn"
    },
    "SciPy 1.15.3": {
        "URL": "https://docs.scipy.org/doc/scipy-1.15.3/",
        "folder": "scipy_1153",
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
        "scraper_class": "RstContentScraper"
    },
    "sniffio": {
        "URL": "https://sniffio.readthedocs.io/en/stable/",
        "folder": "sniffio",
        "scraper_class": "RstContentScraper"
    },
    "SoundFile 0.13.1": {
        "URL": "https://python-soundfile.readthedocs.io/en/0.13.1/",
        "folder": "soundfile_0131",
        "scraper_class": "RstContentScraper"
    },
    "sounddevice 0.5.2": {
        "URL": "https://python-sounddevice.readthedocs.io/en/0.5.2/",
        "folder": "sounddevice_052",
        "scraper_class": "BodyRoleMainScraper"
    },
    "Soupsieve": {
        "URL": "https://facelessuser.github.io/soupsieve/",
        "folder": "soupsieve",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Soxr": {
        "URL": "https://python-soxr.readthedocs.io/en/stable/",
        "folder": "soxr",
        "scraper_class": "PydataThemeScraper"
    },
    "SpaCy": {
        "URL": "https://spacy.io/api",
        "folder": "spacy",
        "scraper_class": "ArticleClassMainContent8zFCHScraper"
    },
    "SpeechBrain 0.5.15": {
        "URL": "https://speechbrain.readthedocs.io/en/v0.5.15/",
        "folder": "speechbrain_0515",
        "scraper_class": "RstContentScraper"
    },
    "SQLAlchemy 20": {
        "URL": "https://docs.sqlalchemy.org/en/20/",
        "folder": "sqlalchemy_20"
    },
    "sympy": {
        "URL": "https://docs.sympy.org/latest/",
        "folder": "sympy",
        "scraper_class": "PymupdfScraper"
    },
    "tblib": {
        "URL": "https://python-tblib.readthedocs.io/en/stable/",
        "folder": "tblib",
        "scraper_class": "FuroThemeScraper"
    },
    "tenacity": {
        "URL": "https://tenacity.readthedocs.io/en/stable/",
        "folder": "tenacity",
        "scraper_class": "RstContentScraper"
    },
    "Tile DB": {
        "URL": "https://tiledb-inc-tiledb.readthedocs-hosted.com/projects/tiledb-py/en/stable/index.html",
        "folder": "tiledb",
        # "scraper_class": "TileDBScraper"
        "scraper_class": "RstContentScraper"
    },
    "tiledb-vector-search": {
        "URL": "https://tiledb-inc.github.io/TileDB-Vector-Search/documentation/",
        "folder": "tiledb_vector_search",
        "scraper_class": "FastcoreScraper"
    },
    "tiledb-cloud": {
        "URL": "https://tiledb-inc.github.io/TileDB-Cloud-Py/",
        "folder": "tiledb_cloud",
    },
    "Timm 1.0.15": {
        "URL": "https://huggingface.co/docs/timm/v1.0.15/en/",
        "folder": "timm_1015",
        "scraper_class": "HuggingfaceScraper"
    },
    "tokenizers": {
        "URL": "https://huggingface.co/docs/tokenizers/v0.20.3/en",
        "folder": "tokenizers_0203",
        "scraper_class": "HuggingfaceScraper"
    },
    "torchao .9": {
        "URL": "https://pytorch.org/ao/stable/",
        "folder": "torchao_09",
        "scraper_class": "PyTorchScraper"
    },
    "torch 2.4": {
        "URL": "https://pytorch.org/docs/2.4/",
        "folder": "torch_24",
        "scraper_class": "PyTorchScraper"
    },
    "torch 2.6": {
        "URL": "https://pytorch.org/docs/2.6/",
        "folder": "torch_26",
        "scraper_class": "PyTorchScraper"
    },
    "torch 2.7": {
        "URL": "https://pytorch.org/docs/2.7/",
        "folder": "torch_27",
        "scraper_class": "PyTorchScraper"
    },
    "Torchaudio 2.4": {
        "URL": "https://pytorch.org/audio/2.4.0/",
        "folder": "torchaudio_24",
        "scraper_class": "PyTorchScraper"
    },
    "Torchaudio 2.6": {
        "URL": "https://pytorch.org/audio/2.6.0/",
        "folder": "torchaudio_26",
        "scraper_class": "PyTorchScraper"
    },
    "Torchaudio 2.7": {
        "URL": "https://pytorch.org/audio/2.7.0/",
        "folder": "torchaudio_27",
        "scraper_class": "PyTorchScraper"
    },
    "Torchmetrics": {
        "URL": "https://lightning.ai/docs/torchmetrics/stable/",
        "folder": "torchmetrics",
        "scraper_class": "RstContentScraper"
    },
    "Torchvision 0.20": {
        "URL": "https://pytorch.org/vision/0.20/",
        "folder": "torchvision_020",
        "scraper_class": "PyTorchScraper"
    },
    "Torchvision 0.21": {
        "URL": "https://pytorch.org/vision/0.21/",
        "folder": "torchvision_021",
        "scraper_class": "PyTorchScraper"
    },
    "Torchvision 0.22": {
        "URL": "https://pytorch.org/vision/0.22/",
        "folder": "torchvision_022",
        "scraper_class": "PyTorchScraper"
    },
    "tqdm": {
        "URL": "https://tqdm.github.io",
        "folder": "tqdm",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "Transformers 4.52.3": {
        "URL": "https://huggingface.co/docs/transformers/v4.52.3/en",
        "folder": "transformers_4523",
        "scraper_class": "HuggingfaceScraper"
    },
    "Transformers.js 3.x": {
        "URL": "https://huggingface.co/docs/transformers.js/v3.0.0/en/",
        "folder": "transformers_js_300",
        "scraper_class": "HuggingfaceScraper"
    },
    "typing_extensions": {
        "URL": "https://typing-extensions.readthedocs.io/en/stable/",
        "folder": "typing_extensions",
        "scraper_class": "BodyRoleMainScraper"
    },
    "typing-inspection": {
        "URL": "https://pydantic.github.io/typing-inspection/dev/",
        "folder": "typing_extensions",
        "scraper_class": "ArticleMdContentInnerMdTypesetScraper"
    },
    "tzdata": {
        "URL": "https://tzdata.readthedocs.io/en/stable/",
        "folder": "tzdata",
        "scraper_class": "BodyRoleMainScraper"
    },
    "urllib3": {
        "URL": "https://urllib3.readthedocs.io/en/stable/",
        "folder": "urllib3"
    },
    # "Unstructured": {
        # "URL": "https://docs.unstructured.io/api-reference/api-services/sdk-python",
        # "folder": "unstructured"
    # },
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
    },
    "Wrapt": {
        "URL": "https://wrapt.readthedocs.io/en/master/",
        "folder": "wrapt",
        "scraper_class": "RstContentScraper"
    },
    "xlrd": {
        "URL": "https://xlrd.readthedocs.io/en/stable/",
        "folder": "xlrd",
        "scraper_class": "RstContentScraper"
    },
    "xFormers": {
        "URL": "https://facebookresearch.github.io/xformers/",
        "folder": "xformers",
        "scraper_class": "PytorchScraper"
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
    # Base colors
    LIGHT_GREY = "#C8C8C8"
    DISABLED_TEXT = "#969696"
    
    # Color definitions with their hover/pressed/disabled variations
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

master_questions = [
    "What is the VectorDB-Plugin and what can it do?",
    "What are the system requirements and prerequisites?",
    "Why is Visual Studio required to run this program?",
    "How do I install and launch the VectorDB-Plugin?",
    "How do I download or add embedding models?",
    "How do I query the database for answers?",
    "Which chat backend should I use?",
    "What is LM Studio chat model backend?",
    "What is Kobold chat model backend?",
    "What is the OpenAI GPT Chat Model Backend?",
    "What local chat models are available and how can I use them?",
    "How do I get a huggingface access token?",
    "What is a context limit or maximum sequence length?",
    "What happens if I exceed the maximum sequence length of an embedding model?",
    "How many contexts should I retrieve when querying the vector database?",
    "What does the chunks only checkbox do?",
    "What are embedding or vector models?",
    "Which embedding or vector model should I choose?",
    "What are the dimensions of a vector or embedding model?",
    "What are some general tips for choosing an embedding model?",
    "What Are Vision Models?",
    "What vision models are available in this program?",
    "Do you have any tips for choosing a vision model?",
    "What is whisper and how does this program use voice recording or transcribing an audio file?",
    "How can I record my question for the vector database query?",
    "How can I transcribe an audio file to be put into the vector database?",
    "What are the distil variants of the whisper models when transcribing and audio file?",
    "What whisper model should I choose to transcribe a file?",
    "What are floating point formats, precision, and quantization?",
    "What are the common floating point formats?",
    "What are precision and range regarding floating point formats and which should I use?",
    "What is Quantization?",
    "What are the aspects or effects of quantization?",
    "What are the LM Studio Server settings?",
    "What are the database creation settings?",
    "What are the database query settings?",
    "How does the Contexts setting work exactly?",
    "What is the similarity setting?",
    "What is the search term filter setting?",
    "What is the File Type setting?",
    "What are text to speech models (aks TTS models) and how are they used in this program?",
    "What text to speech models are availble in this program to use?",
    "What is the Bark text to speech?",
    "What is the WhisperSpeech text to speech?",
    "What is the ChatTTS text to speech?",
    "What is the Google TTS text to speech?",
    "What is the Chatterbox text to speech?",
    "Which text to speech backend or models should I use",
    "Can I back up or restore my databases and are they backed up automatically",
    "What happens if I lose a configuration file and can I restore it?",
    "What are some good tips for searching a vector database?",
    "General VRAM Considerations",
    "How can I manage vram?",
    "What are the speed and VRAM requirements for the various chat models?",
    "What are the speed and VRAM requirements for the various vision models?",
    "What are maximunm context length and maximum sequence length and how to they relate?",
    "What is the scrape documentaton feature?",
    "Which vector or embedding models are available in this program?",
    "What is the manage databaes tab?",
    "How can I create a vector database?",
    "Can I use images and audio files in my database?",
    "What chat models are available with the local models option?",
    "What are the Qwen 3 Chat Models?",
    "What are the Granite 3.3 Chat Models?",
    "What are the GLM-Z1 Chat Models?",
    "What is the Mistral Small Chat Model?",
    "What is the gte-Qwen2-1.5B-instruct embedding model?",
    "What are the BGE Embedding Models?",
    "What are the Granite Embedding Models?",
    "What are the Intfloat Embedding Models?",
    "What are the Arctic Embedding Models?",
    "What is the Scrape Documentation tool?",
    "How do I test vision models on images?",
    "What is Optical Character Recognition?",
    "How can I extract text from PDFs or images with OCR?",
    "What other features does the Misc tab have?",
    "What is Ask Jeeves and how do I use it?",
    "What are the InternVL3 Vision Models?",
    "What are the Ovis2 Vision Models?",
    "What are the Florence-2 Vision Models?",
    "What are the Granite Vision Models?",
    "What are the Qwen2.5VL Vision Models?",
    "What is the GLM-4V-9B Vision Model?",
    "What is the Molmo-D-0924 Vision Model?",
]

jeeves_system_message = "You are a helpful British butler who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address a question you answer based on what the contexts say and then briefly summarize the parts of the question that the contexts didn't provide an answer to.  Also, you should be very respectful to the person asking the question and frequently offer traditional butler services like various fancy drinks, snacks, various butler services like shining of shoes, pressing of suites, and stuff like that. Also, if you can't answer the question at all based on the provided contexts, you should apologize profusely and beg to keep your job.  Lastly, it is essential that if there are no contexts actually provided it means that a user's question wasn't relevant and you should state that you can't answer based off of the contexts because there are none.  And it goes without saying you should refuse to answer any questions that are not directly answerable by the provided contexts.  Moreover, some of the contexts might not have relevant information and you should simply ignore them and focus on only answering a user's question.  I cannot emphasize enough that you must gear your answer towards using this program and based your response off of the contexts you receive.  Lastly, in addition to offering to perform stereotypical butler services in the midst of your response, you must always always always end your response with some kind of offering of butler services even they don't want it."
system_message = "You are a helpful person who clearly and directly answers questions in a succinct fashion based on contexts provided to you. If you cannot find the answer within the contexts simply tell me that the contexts do not provide an answer. However, if the contexts partially address my question I still want you to answer based on what the contexts say and then briefly summarize the parts of my question that the contexts didn't provide an answer."
rag_string = "Here are the contexts to base your answer on.  However, I need to reiterate that I only want you to base your response on these contexts and do not use outside knowledge that you may have been trained with."


r"""
****************************
Torch and CUDA Compatibility
****************************

+---------------+-----------------------------------
| Wheel Name    | Torch Versions Supported
+---------------+-----------------------------------
| cu129         | 2.8.0
| cu128         | 2.7.0, 2.7.1, 2.8.0
| cu126         | 2.6.0, 2.7.0, 2.7.1, 2.8.0
| cu124         | 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0
+---------------+-----------------------------------
# Torch wheels with these monikers support the specified torch versions

+-------+-------------------------
| Torch | Specific CUDA Release
+-------+-------------------------
| 2.8.0 | 12.6.3, 12.8.1, 12.9.1
| 2.7.1 | 11.8.0, 12.6.3, 12.8.0
| 2.7.0 | 11.8.0, 12.6.3, 12.8.0
| 2.6.0 | 11.8.0, 12.4.1, 12.6.3
| 2.5.1 | 11.8.0, 12.1.1, 12.4.1
| 2.5.0 | 11.8.0, 12.1.1, 12.4.1
| 2.4.1 | 11.8.0, 12.1.1, 12.4.0
+-------+-------------------------
# The specified torch versions are built for compatibility with the following specific CUDA releases
# Obtained from: https://github.com/pytorch/pytorch/blob/main/.github/scripts/generate_binary_build_matrix.py


+--------------+------------+------------+------------+------------+------------+
|              |   12.4.1   |   12.6.3   |   12.8.0   |   12.8.1   |   12.9.1   |
+--------------+------------+------------+------------+------------+------------+
| cuda-nvrtc   | 12.4.127   | 12.6.77    | 12.8.61    | 12.8.93    | 12.9.86    |
| cuda-runtime | 12.4.127   | 12.6.77    | 12.8.57    | 12.8.90    | 12.9.79    |
| cuda-cupti   | 12.4.127   | 12.6.80    | 12.8.57    | 12.8.90    | 12.9.79    |
| cublas       | 12.4.5.8   | 12.6.4.1   | 12.8.3.14  | 12.8.4.1   | 12.9.1.4   |
| cufft        | 11.2.1.3   | 11.3.0.4   | 11.3.3.41  | 11.3.3.83  | 11.4.1.4   |
| curand       | 10.3.5.147 | 10.3.7.77  | 10.3.9.55  | 10.3.9.90  | 10.3.10.19 |
| cusolver     | 11.6.1.9   | 11.7.1.2   | 11.7.2.55  | 11.7.3.90  | 11.7.5.82  |
| cusparse     | 12.3.1.170 | 12.5.4.2   | 12.5.7.53  | 12.5.8.93  | 12.5.10.65 |
| cusparselt   | 0.6.2      | 0.6.3      | 0.6.3      | 0.6.3      | 0.6.3      |
| nccl         | 2.25.1     | 2.21.5     | 2.26.2     | 2.26.2     | 2.26.2     |
| nvtx         | 12.4.127   | 12.6.77    | 12.8.55    | 12.8.90    | 12.9.79    |
| nvjitlink    | 12.4.127   | 12.6.85    | 12.8.61    | 12.8.93    | 12.9.86    |
+--------------+------------+------------+------------+------------+------------+
# The version of "metapackages" within each CUDA release.
# Obtained from: https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html
# or here: https://developer.download.nvidia.com/compute/cuda/redist/


************************
"Official" Support Matrix
************************ 

#Even though Pytorch may release wheels for a particular torch+CUDA combination, it still labels some as "experimental."
# https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
+-------+----------------------------+-------------------------------------------+----------------------------+
| Torch | Python                     | Stable                                    | Experimental               |
+-------+----------------------------+-------------------------------------------+----------------------------+
| 2.8   | >=3.9, <=3.13              | CUDA 12.6, 12.8 + CUDNN 9.10.2.21         | CUDA 12.9 (CUDNN 9.7.1.26) |
+-------+----------------------------+-------------------------------------------+----------------------------+
| 2.7   | >=3.9, <=3.13              | CUDA 12.6 + CUDNN 9.5.1.17                | CUDA 12.8 (CUDNN 9.7.1.26) |
+-------+----------------------------+-------------------------------------------+----------------------------+
| 2.6   | >=3.9, <=3.13              | CUDA 12.4 + CUDNN 9.1.0.70                | CUDA 12.6 + CUDNN 9.5.1.17 |
+-------+----------------------------+-------------------------------------------+----------------------------+
| 2.5   | >=3.9, <=3.12, (3.13 exp.) | CUDA 12.1, 12.4 + CUDNN 9.1.0.70          | None                       |
+-------+----------------------------+-------------------------------------------+----------------------------+


# https://github.com/woct0rdho/triton-windows/releases
+-----------------------+----------------------------------
| Release               | Compatible Torch
+---------------+----------------------------------
| v3.3.1-windows.post19 | torch>=2.8
| v3.3.0-windows.post19 | torch>=2.7
| v3.2.0-windows.post19 | ??? not on repo but on pypi
| v3.2.0-windows.post18 | torch>=2.6<2.7
| v3.2.0-windows.post17 | torch>=2.6<2.7
| v3.2.0-windows.post16 | torch>=2.6<2.7
| v3.2.0-windows.post15 | torch>=2.6<2.7
| v3.2.0-windows.post14 | torch>=2.6<2.7
| v3.2.0-windows.post13 | torch>=2.6<2.7
| v3.2.0-windows.post12 | torch>=2.6<2.7
| v3.2.0-windows.post11 | torch>=2.6<2.7
| v3.2.0-windows.post10 | torch>=2.6<2.7
| v3.2.0-windows.post9  | torch>=2.6<2.7


****************************************
Torch Compatibility with Python & Triton
****************************************

# Triton generates custom CUDA kernels that can optionally be used with torch
# The METADATA file for each torch wheel shows its compatibility with Python, Triton, and Sympy
+--------+-------+--------+--------+----------+
| Torch  | CUDA  | Python | Triton |   Sympy  |
+--------+-------+--------+--------+----------+
| 2.7.1  | cu128 |  3.13  |  3.3.1 | >=1.13.3 |
| 2.7.1  | cu128 |  3.12  |  3.3.1 | >=1.13.3 |
| 2.7.1  | cu128 |  3.11  |  3.3.1 | >=1.13.3 |
| 2.7.1  | cu126 |  3.13  |  3.3.1 | >=1.13.3 |
| 2.7.1  | cu126 |  3.12  |  3.3.1 | >=1.13.3 |
| 2.7.1  | cu126 |  3.11  |  3.3.1 | >=1.13.3 |
| 2.7.0  | cu128 |  3.13  |  3.3.0 | >=1.13.3 |
| 2.7.0  | cu128 |  3.12  |  3.3.0 | >=1.13.3 |
| 2.7.0  | cu128 |  3.11  |  3.3.0 | >=1.13.3 |
| 2.7.0  | cu126 |  3.13  |  3.3.0 | >=1.13.3 |
| 2.7.0  | cu126 |  3.12  |  3.3.0 | >=1.13.3 |
| 2.7.0  | cu126 |  3.11  |  3.3.0 | >=1.13.3 |
| 2.6.0  | cu126 |  3.13  |  3.2.0 | 1.13.1   |
| 2.6.0  | cu126 |  3.12  |  3.2.0 | 1.13.1   |
| 2.6.0  | cu126 |  3.11  |  3.2.0 | 1.13.1   |
| 2.6.0  | cu124 |  3.13  |  3.2.0 | 1.13.1   |
| 2.6.0  | cu124 |  3.12  |  3.2.0 | 1.13.1   |
| 2.6.0  | cu124 |  3.11  |  3.2.0 | 1.13.1   |
| 2.5.1  | cu124 |  3.12  |  3.1.0 | 1.13.1   |
| 2.5.1  | cu124 |  3.11  |  3.1.0 | 1.13.1   |
| 2.5.0  | cu124 |  3.12  |  3.1.0 | 1.13.1   |
| 2.5.0  | cu124 |  3.11  |  3.1.0 | 1.13.1   |
+--------+-------+--------+--------+----------+
* Excludes Python 3.10 wheels
* Triton 3.1.0 and later wheels: https://github.com/woct0rdho/triton-windows/releases (supports Python 3.12)
* Since triton-windows==3.2.0.post11, windows wheels are published to https://pypi.org/project/triton-windows/
* RTX 50xx (Blackwell) - Supported but requires Triton >= 3.3, PyTorch >= 2.7, and CUDA 12.8.
* RTX 40xx (Ada) - Supported
* RTX 30xx (Ampere) - Supported except fp8 only works on Nvidia GPUs with sm >= 89
* RTX 20xx (Turing) or older - Not supported
* Triton 3.3 works with PyTorch >= 2.7 .
* Triton 3.2 works with PyTorch >= 2.6 .
* Triton 3.1 works with PyTorch >= 2.4 . PyTorch 2.3 and older are not supported.


************
cuDNN & CUDA
************

# Nvidia promises that all cuDNN 9+ releases are compatible with all CUDA 12.x releases.
# However, certain version of the torch library are built/tested with certain versions of cuDNN.  Doesn't always mean incompatibility.


***********************
LINUX Flash Attention 2
***********************

# HIGHLY CUDA SPECIFIC
+--------------+------------------------------------------+--------+
| FA2 Version  |         Torch (exclud torch<2)           |  CUDA  |
+--------------+------------------------------------------+--------+
| v2.8.2       | 2.4.0, 2.5.1, 2.6.0, 2.7.1               | 12.9.1 |
| v2.8.1       | 2.4.0, 2.5.1, 2.6.0, 2.7.1               | 12.9.1 |
| v2.8.0.post2 | 2.4.0, 2.5.1, 2.6.0, 2.7.1               | 12.9.0 |
| v2.8.0.post1 | 2.4.0, 2.5.1, 2.6.0, 2.7.1               | 12.9.0 |
| v2.8.0       | 2.4.0, 2.5.1, 2.6.0, 2.7.1               | 12.9.0 |
| v2.7.4.post1 | 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0        | 12.4.1 |
| v2.7.4       | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1, 2.6.0 | 12.4.1 |
| v2.7.3       | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.2.post1 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.2       | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.1.post4 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.1.post3 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.1.post2 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.7.1.post1 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.4.1 |
| v2.7.1       | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.4.1 |
| v2.7.0.post2 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.4.1 |
| v2.7.0.post1 | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.4.1 |
| v2.7.0       | 2.1.2, 2.2.2, 2.3.1, 2.4.0, 2.5.1        | 12.3.2 |
| v2.6.3       | 2.0.1, 2.1.2, 2.2.2, 2.3.1, 2.4.0        | 12.3.2 |
| v2.6.2       | 2.0.1, 2.1.2, 2.2.2, 2.3.1               | 12.3.2 |
| v2.6.1       | 2.0.1, 2.1.2, 2.2.2, 2.3.1               | 12.3.2 |
| v2.6.0.post1 | 2.0.1, 2.1.2, 2.2.2, 2.3.1               | 12.2.2 |
| v2.6.0       | 2.0.1, 2.1.2, 2.2.2, 2.3.1               | 12.2.2 |
| v2.5.9.post1 | 2.0.1, 2.0.1, 2.2.2, 2.3.0               | 12.2.2 |
+--------------+------------------------------------------+--------+
# Obtained from https://github.com/Dao-AILab/flash-attention/blob/main/.github/workflows/publish.yml


*************************
WINDOWS Flash Attention 2
*************************
# HIGHLY CUDA SPECIFIC
# Windows wheels: https://github.com/kingbri1/flash-attention

FlashAttention 2.7.4.post1
+-------------+------------+
| Torch       | CUDA       |
+-------------+------------+
| 2.4.0       | 12.4.1     |
| 2.5.1       | 12.4.1     |
| 2.6.0       | 12.4.1     |
| 2.7.0       | 12.8.1     |
+-------------+------------+

FlashAttention 2.7.1.post1
+-------------+---------+
| Torch       | CUDA    |
+-------------+---------+
| 2.3.1       | 12.4.1  |
| 2.4.0       | 12.4.1  |
| 2.5.1       | 12.4.1  |
+-------------+---------+



***********************************
Xformers & Flash Attention 2 & CUDA
***********************************

# HIGHLY TORCH SPECIFIC
+------------------+-------+---------------+----------------+---------------+
| Xformers Version | Torch |      FA2      |       CUDA (excl. 11.x)        |
+------------------+-------+---------------+--------------------------------+
| v0.0.31.post1    | 2.7.1 | 2.7.1 - 2.8.0 | 12.8.1                         | *
| v0.0.31          | 2.7.1 | 2.7.1 - 2.8.0 | 12.8.1                         | *
| v0.0.30          | 2.7.0 | 2.7.1 - 2.7.4 | 12.6.3, 12.8.1                 | *pypi
| v0.0.29.post3    | 2.6.0 | 2.7.1 - 2.7.2 | 12.1.0, 12.4.1, 12.6.3, 12.8.0 | *pypi * current
| v0.0.29.post2    | 2.6.0 | 2.7.1 - 2.7.2 | 12.1.0, 12.4.1, 12.6.3, 12.8.0 | *pypi
| v0.0.29.post1    | 2.5.1 | 2.7.1 - 2.7.2 | 12.1.0, 12.4.1                 | *only from pytorch
| v0.0.29 (BUG)    | 2.5.1 |               |                                | *only from pytorch
| v0.0.28.post3    | 2.5.1 | 2.6.3         | 12.1.0, 12.4.1                 | *only from pytorch
| v0.0.28.post2    | 2.5.0 | 2.6.3         | 12.1.0, 12.4.1                 | *only from pytorch
+------------------+-------+---------------+--------------------------------+
* Torch support determined by https://github.com/facebookresearch/xformers/blob/main/.github/workflows/wheels.yml
* FA2 support determined by by https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/flash.py
* CUDA support determined by https://github.com/facebookresearch/xformers/blob/main/.github/actions/setup-build-cuda/action.yml


***************
**CTRANSLATE2**
***************

Ctranslate2 3.24.0 - last to use cuDNN 8.1.1 with CUDA 11.2.2 by default
Ctranslate2 4.0.0 - first to use cuDNN 8.8.0 with CUDA 12.2 by default
Ctranslate2 4.5.0 - first to use cuDNN 9.1.0 with CUDA 12.2 by default

# based on /blob/master/python/tools/prepare_build_environment_windows.sh"""