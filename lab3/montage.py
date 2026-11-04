import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

INSTANCES = ["TSPA", "TSPB"]

GROUPS = [
    ("constructive", "Constructive heuristics (lab 1)",
     ["random", "nn_end", "nn_any", "greedy_cycle"]),
    ("regret", "Regret heuristics (lab 2)",
     ["nn_any_regret", "greedy_cycle_regret",
      "nn_any_weighted", "greedy_cycle_weighted",
      "nn_any_weighted_1_2", "greedy_cycle_weighted_1_2",
      "nn_any_weighted_2_1", "greedy_cycle_weighted_2_1"]),
    ("local_search", "Local search (lab 3)",
     ["steepest_nodes_random", "steepest_nodes_greedy",
      "steepest_edges_random", "steepest_edges_greedy",
      "greedy_nodes_random", "greedy_nodes_greedy",
      "greedy_edges_random", "greedy_edges_greedy"]),
]

COLS = 2
PAGE_W, PAGE_H = 8.27, 11.69

os.makedirs("montages", exist_ok=True)

for inst in INSTANCES:
    for key, title, methods in GROUPS:
        rows = (len(methods) + COLS - 1) // COLS
        fig, axes = plt.subplots(rows, COLS, figsize=(PAGE_W, PAGE_H * rows / 4 if rows < 4 else PAGE_H))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax in axes:
            ax.axis("off")
        for ax, m in zip(axes, methods):
            path = f"plots/{inst}_{m}.png"
            if os.path.exists(path):
                ax.imshow(mpimg.imread(path))
            else:
                ax.text(0.5, 0.5, "missing: " + m, ha="center", va="center")
        fig.suptitle(f"{inst} - {title}", fontsize=12, y=0.995)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, wspace=0.02, hspace=0.02)
        fig.savefig(f"montages/{inst}_{key}.png", dpi=200)
        plt.close(fig)

print("done: folder montages")