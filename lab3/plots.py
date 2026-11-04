import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTANCES = ["TSPA", "TSPB"]
TITLES = {
    "random": "Random solution",
    "nn_end": "Nearest neighbor (end only)",
    "nn_any": "Nearest neighbor (any position)",
    "greedy_cycle": "Greedy cycle",
    "nn_any_regret": "NN any + 2-regret",
    "greedy_cycle_regret": "Greedy cycle + 2-regret",
    "nn_any_weighted": "NN any + weighted (1:1)",
    "greedy_cycle_weighted": "Greedy cycle + weighted (1:1)",
    "nn_any_weighted_1_2": "NN any + weighted (1:2)",
    "greedy_cycle_weighted_1_2": "Greedy cycle + weighted (1:2)",
    "nn_any_weighted_2_1": "NN any + weighted (2:1)",
    "greedy_cycle_weighted_2_1": "Greedy cycle + weighted (2:1)",
    "steepest_nodes_random": "Steepest LS, nodes exchange, random start",
    "steepest_nodes_greedy": "Steepest LS, nodes exchange, greedy start",
    "steepest_edges_random": "Steepest LS, edges exchange, random start",
    "steepest_edges_greedy": "Steepest LS, edges exchange, greedy start",
    "greedy_nodes_random": "Greedy LS, nodes exchange, random start",
    "greedy_nodes_greedy": "Greedy LS, nodes exchange, greedy start",
    "greedy_edges_random": "Greedy LS, edges exchange, random start",
    "greedy_edges_greedy": "Greedy LS, edges exchange, greedy start",
}


def load(name):
    pts = []
    for line in open(f"../data/{name}.csv"):
        p = line.replace(";", " ").replace(",", " ").split()
        if len(p) >= 3:
            pts.append((float(p[0]), float(p[1]), int(float(p[2]))))
    return pts


def dist(a, b):
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def obj(pts, s):
    total = 0
    for i in range(len(s)):
        a = pts[s[i]]
        b = pts[s[(i + 1) % len(s)]]
        total += a[2] + dist(a, b)
    return total


os.makedirs("plots", exist_ok=True)

for name in INSTANCES:
    pts = load(name)
    costs = [p[2] for p in pts]
    vmin, vmax = min(costs), max(costs)
    for m, title in TITLES.items():
        path = f"results/best_{name}_{m}.txt"
        if not os.path.exists(path):
            print("brak pliku:", path)
            continue
        sol = [int(x) for x in open(path).read().split()]
        chosen = set(sol)

        fig, ax = plt.subplots(figsize=(7, 5.2))

        rest = [p for i, p in enumerate(pts) if i not in chosen]
        ax.scatter([p[0] for p in rest], [p[1] for p in rest],
                   s=12, c="lightgrey", zorder=1)

        ax.plot([pts[i][0] for i in sol] + [pts[sol[0]][0]],
                [pts[i][1] for i in sol] + [pts[sol[0]][1]],
                color="black", lw=0.8, zorder=2)

        sc = ax.scatter(
            [pts[i][0] for i in sol],
            [pts[i][1] for i in sol],
            c=[pts[i][2] for i in sol],
            s=[20 + 80 * (pts[i][2] - vmin) / (vmax - vmin) for i in sol],
            cmap="viridis", vmin=vmin, vmax=vmax,
            edgecolors="black", linewidths=0.4, zorder=3,
        )
        fig.colorbar(sc, ax=ax, label="node cost")

        ax.set_title(f"{name} - {title}\nobjective = {obj(pts, sol)}", fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(f"plots/{name}_{m}.png", dpi=130)
        plt.close(fig)

print("done")