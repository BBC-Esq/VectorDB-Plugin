from charts.gpu_info import GPUS

VRAM_SIZES = sorted({info["memory_size_gb"] for info in GPUS.values()})


def gpu_rows(min_vram, max_vram):
    rows = [
        {
            "name": name,
            "vram": info["memory_size_gb"],
            "memory": info["memory_type"],
            "cores": info["cuda_cores"],
            "tensor": info["tensor_cores"],
            "arch": info["architecture"],
            "cc": f'{info["cuda_major_version"]}.{info["cuda_minor_version"]}',
            "fp16": round(info["half_float_performance_gflop_s"] / 1000, 1),
            "released": info["release_date"].year if info.get("release_date") else None,
        }
        for name, info in GPUS.items()
        if min_vram <= info["memory_size_gb"] <= max_vram
    ]
    rows.sort(key=lambda r: r["cores"], reverse=True)
    return rows
