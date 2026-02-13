import matplotlib.pyplot as plt

def create_vector_models_comparison_plot():
    models_data = [
        ("bge-small-en-v1.5", 529.34, 1058.68),
        ("e5-small-v2", 542.37, 1084.74),
        ("Granite-30m-English", 664.24, 1328.48),
        ("e5-base-v2", 779.93, 1559.86),
        ("Granite-125m-English", 885.93, 1771.86),
        ("bge-base-en-v1.5", 911.93, 1823.86),
        ("e5-large-v2", 1428.12, 2856.24),
        ("bge-large-en-v1.5", 1512.24, 3024.48),
        ("arctic-embed-m-v2.0", 1785.93, 3571.86),
        ("arctic-embed-l-v2.0", 2037.93, 4075.86),
        ("Qwen3-Embedding-0.6B", 2974.87, 5949.74),
        ("inf-retriever-v1-1.5b", 6303.24, 12606.48),
        ("Qwen3-Embedding-4B", 9234.05, 18468.10),
        ("Qwen3-Embedding-8B", 15634.17, 31268.34),
        ("inf-retriever-v1-7b", 17274.20, 34548.40),
    ]

    models_data.sort(key=lambda r: r[1])

    names = [r[0] for r in models_data]
    gpu_mb = [r[1] for r in models_data]
    cpu_extra_mb = [r[2] - r[1] for r in models_data]

    plt.rcParams.update({
        "figure.facecolor": "#2e2e2e",
        "axes.facecolor": "#2e2e2e",
        "axes.edgecolor": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "text.color": "white",
        "legend.edgecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = range(len(names))
    ax.barh(y_pos, gpu_mb, color="#4c78a8", label="GPU – half precision (MB)")
    ax.barh(y_pos, cpu_extra_mb, left=gpu_mb,
            color="#f58518", alpha=0.6,
            label="CPU – additional for float32 (MB)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Memory (MB)")
    ax.set_title("Vector-model memory usage\nGPU half-precision vs. CPU float32")

    legend = ax.legend(facecolor="#2e2e2e", framealpha=0.8)
    for text in legend.get_texts():
        text.set_color("white")

    fig.tight_layout()
    return fig
