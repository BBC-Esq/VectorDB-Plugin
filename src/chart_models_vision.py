import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def create_vision_models_comparison_plot():
    model_categories = {
        "florence": {
            "models": [
                # "Florence2-Large (cpu)",
                # "Florence2-Base (cpu)",
            ],
            "color": "#2E8B57",
            "label": "Can run on CPU"
        },
    }

    data = [
        {"model": "GLM-4.1V-9B-Thinking", "cps": 200.96, "memory": 9802.12},
        {"model": "Qwen VL - 3b", "cps": 178.31, "memory": 6306.30},
        {"model": "Qwen VL - 7b", "cps": 173.67, "memory": 9559.89},
        {"model": "Granite Vision - 2b", "cps": 217.64, "memory": 4094.18},
        {"model": "InternVL3 - 1b", "cps": 274.25, "memory": 2318.05},
        {"model": "InternVL3 - 2b", "cps": 244.36, "memory": 3153.87},
        {"model": "InternVL3 - 8b", "cps": 255.95, "memory": 8153.30},
        {"model": "InternVL3 - 14b", "cps": 162.58, "memory": 12998.80},
        {"model": "Liquid-VL - 1.6B", "cps": 437.50, "memory": 1396.00},
        {"model": "Liquid-VL - 450m", "cps": 497.64, "memory": 497.64},
    ]

    df = pd.DataFrame(data)
    df["memory"] = df["memory"] / 1024
    df = df.sort_values(by="memory")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#2e2e2e')
    ax1.set_facecolor('#2e2e2e')
    ax1.set_title("Model Comparison - Memory Usage vs Characters per Second", fontsize=16, color='white', pad=10)

    ax2 = ax1.twinx()
    gradient = LinearSegmentedColormap.from_list("", ["#003328", "#004D40"])

    bars = []
    for i, (index, row) in enumerate(df.iterrows()):
        border_color = None
        border_width = 0
        for category in model_categories.values():
            if row["model"] in category["models"]:
                border_color = category["color"]
                border_width = 3
                break

        bar = ax1.bar(i, row["memory"], color=gradient(i/len(df)), alpha=0.7, 
                      edgecolor=border_color, linewidth=border_width)
        bars.append(bar[0])

    ax1.bar(0, 0, color=gradient(0.5), alpha=0.7, label="Memory Usage")

    ax1.set_xlabel("Model", color="white")
    ax1.set_ylabel("Memory Usage (GB)", color="white", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="white", colors="white")
    ax1.tick_params(axis="x", labelcolor="white", colors="white", rotation=45)

    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df["model"], rotation=45, ha="right")

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', verticalalignment='bottom', color='white', ha='center')

    ax1.grid(True, linestyle='--', alpha=0.1, color='white')

    line = ax2.plot(range(len(df)), df["cps"], color="#5F9EA0", marker="o", label='Characters per Second (cps)')
    ax2.set_ylabel("Characters per Second", color="white", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="white")

    for i, cps in enumerate(df["cps"]):
        ax2.text(i, cps, f'{cps:.2f}', ha='center', va='bottom', color='white')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    category_patches = [Patch(facecolor='none', edgecolor=cat["color"], label=cat["label"], linewidth=2) 
                       for cat in model_categories.values()]
    all_handles = lines1 + lines2 + category_patches
    all_labels = labels1 + labels2 + [cat["label"] for cat in model_categories.values()]

    ax1.legend(all_handles, all_labels, loc='upper center', 
              fancybox=True, shadow=True, ncol=len(all_handles),
              facecolor='#2e2e2e', edgecolor='white', labelcolor='white')

    fig.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.25)

    return fig

if __name__ == "__main__":
    fig = create_vision_models_comparison_plot()
    plt.show()