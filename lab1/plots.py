import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INSTANCES = ["TSPA", "TSPB"]
METHODS = ["random", "nn_end", "nn_any", "greedy_cycle"]
TITLES = {
    "random": "Random solution",
    "nn_end": "Nearest neighbor (end only)",
    "nn_any": "Nearest neighbor (any position)",
    "greedy_cycle": "Greedy cycle",
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
    for m in METHODS:
        sol = [int(x) for x in open(f"results/best_{name}_{m}.txt").read().split()]
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

        ax.set_title(f"{name} - {TITLES[m]}\nobjective = {obj(pts, sol)}", fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(f"plots/{name}_{m}.png", dpi=130)
        plt.close(fig)

print("done")
