import os
import sys
from pathlib import Path


def set_cuda_paths():
    site_packages = Path(sys.executable).parent.parent / 'Lib' / 'site-packages'
    nvidia_base_path = site_packages / 'nvidia'
    cublas_path = nvidia_base_path / 'cublas' / 'bin'
    cudnn_path = nvidia_base_path / 'cudnn' / 'bin'
    paths_to_add = [str(p) for p in (cublas_path, cudnn_path) if p.is_dir()]
    if paths_to_add:
        current_value = os.environ.get('PATH', '')
        new_value = os.pathsep.join(paths_to_add + ([current_value] if current_value else []))
        os.environ['PATH'] = new_value

    triton_cuda_path = site_packages / 'triton' / 'backends' / 'nvidia'
    if triton_cuda_path.is_dir():
        os.environ['CUDA_PATH'] = str(triton_cuda_path)
