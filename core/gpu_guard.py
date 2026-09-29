import os
import subprocess
import sys

from core.constants import MIN_CUDA_COMPUTE_CAPABILITY

_GUARD_ENV = "VECTORDB_GPU_GUARD"


def query_nvidia_gpus():
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    gpus = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            major, minor = (int(x) for x in parts[-1].split("."))
        except ValueError:
            continue
        gpus.append({"index": index, "name": ", ".join(parts[1:-1]), "compute_capability": (major, minor)})
    return gpus


def is_supported_gpu(gpu):
    return gpu["compute_capability"] >= MIN_CUDA_COMPUTE_CAPABILITY


def describe_gpus(gpus):
    return ", ".join(
        f"{g['name']} (compute capability {g['compute_capability'][0]}.{g['compute_capability'][1]})"
        for g in gpus
    )


def hide_unsupported_gpus():
    if os.environ.get(_GUARD_ENV) or "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    os.environ[_GUARD_ENV] = "1"
    gpus = query_nvidia_gpus()
    if not gpus:
        return
    usable = [g for g in gpus if is_supported_gpu(g)]
    if len(usable) == len(gpus):
        return
    unsupported = describe_gpus([g for g in gpus if not is_supported_gpu(g)])
    required = "{}.{}".format(*MIN_CUDA_COMPUTE_CAPABILITY)
    if usable:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g["index"]) for g in usable)
        print(f"\033[93mIgnoring GPU(s) older than compute capability {required}: {unsupported}\033[0m")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print(f"\033[93mThis PyTorch build needs an NVIDIA GPU with compute capability {required} or newer; "
              f"found {unsupported}. Running in CPU mode.\033[0m")
