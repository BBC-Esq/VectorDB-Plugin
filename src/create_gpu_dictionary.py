# processes gpu_database.xlsx within the Assets folder, which is derived from https://github.com/painebenjamin/dbgpu
# creates a new file named gpu_info.py containing a dictionary of all semi-recent gpu info
import sys
from pathlib import Path
from datetime import datetime, date
from typing import TypedDict, Dict
import pandas as pd
from PySide6.QtWidgets import QApplication, QFileDialog

class GPUInfo(TypedDict):
    gpu_name: str
    generation: str
    architecture: str
    release_date: date
    bus_interface: str
    memory_size_gb: int
    memory_type: str
    cuda_cores: int
    streaming_multiprocessors: int
    tensor_cores: int
    cuda_major_version: int
    cuda_minor_version: int
    half_float_performance_gflop_s: int
    single_float_performance_gflop_s: int
    tpu_url: str

def main():
    app = QApplication(sys.argv)

    # 1) Let user pick the Excel file
    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select GPU specs .xlsx",
        "",
        "Excel Files (*.xlsx *.xls)"
    )
    if not file_path:
        print("No file selected. Exiting.")
        return

    # 2) Read into DataFrame
    df = pd.read_excel(file_path)

    # 3) Build the dict
    gpu_dict: Dict[str, GPUInfo] = {}
    for _, row in df.iterrows():
        name = str(row["name"])
        cell = row["release_date"]
        release_dt = pd.to_datetime(cell).date()

        info: GPUInfo = {
            "gpu_name":                   str(row["gpu_name"]),
            "generation":                 str(row["generation"]),
            "architecture":               str(row["architecture"]),
            "release_date":               release_dt,
            "bus_interface":              str(row["bus_interface"]),
            "memory_size_gb":             int(row["memory_size_gb"]),
            "memory_type":                str(row["memory_type"]),
            "cuda_cores":                 int(row["shading_units"]),
            "streaming_multiprocessors":  int(row["streaming_multiprocessors"]),
            "tensor_cores":               int(row["tensor_cores"]),
            "cuda_major_version":         int(row["cuda_major_version"]),
            "cuda_minor_version":         int(row["cuda_minor_version"]),
            "half_float_performance_gflop_s":   int(row["half_float_performance_gflop_s"]),
            "single_float_performance_gflop_s": int(row["single_float_performance_gflop_s"]),
            "tpu_url":                    str(row["tpu_url"]),
        }
        gpu_dict[name] = info

    # 4) Write out gpu_info.py
    out_file = Path.cwd() / "gpu_info.py"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Auto-generated GPU info module\n")
        f.write("from typing import TypedDict, Dict\n")
        f.write("from datetime import date\n\n")
        f.write("class GPUInfo(TypedDict):\n")
        for field, typ in [
            ("gpu_name", "str"),
            ("generation", "str"),
            ("architecture", "str"),
            ("release_date", "date"),
            ("bus_interface", "str"),
            ("memory_size_gb", "int"),
            ("memory_type", "str"),
            ("cuda_cores", "int"),
            ("streaming_multiprocessors", "int"),
            ("tensor_cores", "int"),
            ("cuda_major_version", "int"),
            ("cuda_minor_version", "int"),
            ("half_float_performance_gflop_s", "int"),
            ("single_float_performance_gflop_s", "int"),
            ("tpu_url", "str"),
        ]:
            f.write(f"    {field}: {typ}\n")
        f.write("\nGPUS: Dict[str, GPUInfo] = {\n")
        for name, info in gpu_dict.items():
            f.write(f"    {name!r}: {{\n")
            for k, v in info.items():
                if k == "release_date":
                    if v is None or str(v) == "NaT":
                        f.write("        'release_date': None,\n")
                    else:
                        f.write(f"        'release_date': date.fromisoformat({v.isoformat()!r}),\n")
                elif isinstance(v, str):
                    f.write(f"        {k!r}: {v!r},\n")
                else:
                    f.write(f"        {k!r}: {v!r},\n")
            f.write("    },\n")
        f.write("}\n")

    print(f"gpu_info.py generated at: {out_file}")

if __name__ == "__main__":
    main()
