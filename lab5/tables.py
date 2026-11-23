import csv

NAMES = {
    "random": "Random",
    "nn_end": "NN (end only)",
    "nn_any": "NN (any position)",
    "greedy_cycle": "Greedy cycle",
    "nn_any_regret": "NN any + 2-regret",
    "greedy_cycle_regret": "Greedy cycle + 2-regret",
    "nn_any_weighted": "NN any + weighted (1:1)",
    "greedy_cycle_weighted": "Greedy cycle + weighted (1:1)",
    "nn_any_weighted_1_2": "NN any + weighted (1:2)",
    "greedy_cycle_weighted_1_2": "Greedy cycle + weighted (1:2)",
    "nn_any_weighted_2_1": "NN any + weighted (2:1)",
    "greedy_cycle_weighted_2_1": "Greedy cycle + weighted (2:1)",
    "steepest_nodes_random": "Steepest, nodes, random start",
    "steepest_nodes_greedy": "Steepest, nodes, greedy start",
    "steepest_edges_random": "Steepest, edges, random start (baseline)",
    "steepest_edges_greedy": "Steepest, edges, greedy start",
    "greedy_nodes_random": "Greedy, nodes, random start",
    "greedy_nodes_greedy": "Greedy, nodes, greedy start",
    "greedy_edges_random": "Greedy, edges, random start",
    "greedy_edges_greedy": "Greedy, edges, greedy start",
    "candidate_edges_random": "Steepest, candidate moves (K=10), random start",
    "candidate_edges_greedy": "Steepest, candidate moves (K=10), greedy start",
    "lm_edges_random": "Steepest, list of moves (LM), random start",
    "lm_edges_greedy": "Steepest, list of moves (LM), greedy start",
}
INST = ["TSPA", "TSPB"]

rows = list(csv.DictReader(open("results/results.csv")))
res = {(r["instance"], r["method"]): r for r in rows}
order = []
for r in rows:
    if r["method"] not in order:
        order.append(r["method"])

print("Table 1. Objective function value (min / max / avg)")
print("| Inst. | Method | Min | Max | Avg |")
print("|---|---|---|---|---|")
for inst in INST:
    for m in order:
        r = res[(inst, m)]
        print(f"| {inst} | {NAMES.get(m, m)} | {r['min']} | {r['max']} | {float(r['avg']):.1f} |")

print()
print("Table 2. Running time of one run [ms], average (min - max)")
print("| Method | TSPA | TSPB |")
print("|---|---|---|")
for m in order:
    cells = []
    for inst in INST:
        r = res[(inst, m)]
        cells.append(f"{float(r['time_avg_ms']):.3g} ({float(r['time_min_ms']):.3g} - {float(r['time_max_ms']):.3g})")
    print(f"| {NAMES.get(m, m)} | {cells[0]} | {cells[1]} |")

print()
print("Table 3. Local search: applied moves and evaluated moves per run (averages over 200 runs)")
print("| Method | TSPA iterations | TSPA evaluated moves | TSPB iterations | TSPB evaluated moves |")
print("|---|---|---|---|---|")
stats = {}
for r in csv.DictReader(open("results/ls_stats.csv")):
    stats[(r["instance"], r["method"])] = r
for m in order:
    if (INST[0], m) not in stats:
        continue
    if not any(k in m for k in ("steepest", "greedy_nodes", "greedy_edges", "candidate", "lm_")):
        continue
    a, b = stats[("TSPA", m)], stats[("TSPB", m)]
    print(f"| {NAMES.get(m, m)} | {float(a['avg_iterations']):.1f} | {float(a['avg_evaluated_moves']):.0f} | "
          f"{float(b['avg_iterations']):.1f} | {float(b['avg_evaluated_moves']):.0f} |")