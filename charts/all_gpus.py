import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from gpu_info import GPUS

def create_gpu_comparison_plot(min_vram_threshold=6, max_vram_threshold=8):
    filtered = {name: info for name, info in GPUS.items() if min_vram_threshold <= info["memory_size_gb"] <= max_vram_threshold}
    sorted_nvidia = sorted(filtered.items(), key=lambda item: item[1]["cuda_cores"], reverse=True)
    names = [name for name, _ in sorted_nvidia]
    compute_units = [info["cuda_cores"] for _, info in sorted_nvidia]
    sizes = [info["memory_size_gb"] for _, info in sorted_nvidia]
    gradient_nvidia = LinearSegmentedColormap.from_list("", ["#003328", "#00CC66"])
    plt.rcParams["figure.autolayout"] = True
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#4A4A4A")
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor("#4A4A4A")
    bars = ax1.barh(names, compute_units, color=gradient_nvidia(np.linspace(0, 1, len(compute_units))), label="NVIDIA CUDA Cores")
    max_units = max(compute_units) if compute_units else 0
    for i, bar in enumerate(bars):
        pct = (compute_units[i] / max_units) * 100 if max_units else 0
        ax1.text(150, bar.get_y() + bar.get_height() / 2, f"{compute_units[i]:,} - {pct:.2f}%", va="center", ha="left", color="white", fontsize=10)
    ax1.set_xlabel("CUDA Cores", color="white")
    ax1.set_ylabel("Graphics Cards", color="white", labelpad=15)
    ax1.set_title(f"Graphics Cards: CUDA Cores and VRAM Comparison ({min_vram_threshold}GB <= VRAM <= {max_vram_threshold}GB)", color="white", pad=20)
    ax1.tick_params(axis="both", colors="white")
    ax2 = ax1.twiny()
    ax2.plot(sizes, names, "o-", color="orange", label="VRAM (GB)")
    ax2.set_xlabel("VRAM (GB)", color="white")
    ax2.xaxis.set_label_position("bottom")
    ax2.xaxis.tick_bottom()
    ax2.tick_params(axis="x", colors="white")
    ax1.xaxis.set_label_position("top")
    ax1.xaxis.tick_top()
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=gradient_nvidia(0.5), edgecolor="none", label="NVIDIA CUDA Cores"),
        plt.Line2D([0], [0], color="orange", marker="o", linestyle="-", label="VRAM (GB)")
    ]
    ax2.legend(handles=legend_elements, loc="upper right", facecolor="#4A4A4A", edgecolor="white", labelcolor="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("white")
    vram_lines = [2, 4, 6, 8, 10, 11, 12, 16, 20, 24, 32]
    for vram_value in vram_lines:
        if vram_value in sizes:
            ax2.axvline(x=vram_value, color="#A8A8A8", linestyle="--", linewidth=0.5)
    ax2.set_xticks(vram_lines)
    ax2.set_xlim(0, 33)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1)
    return fig

if __name__ == "__main__":
    fig = create_gpu_comparison_plot(12, 24)
    plt.show()
