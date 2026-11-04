import csv
import glob
import math
import os

from openpyxl import load_workbook
from openpyxl.styles import Font

INSTANCES = ["TSPA", "TSPB"]
METHODS = [
    "random", "nn_end", "nn_any", "greedy_cycle",
    "nn_any_regret", "greedy_cycle_regret",
    "nn_any_weighted", "greedy_cycle_weighted",
    "nn_any_weighted_1_2", "greedy_cycle_weighted_1_2",
    "nn_any_weighted_2_1", "greedy_cycle_weighted_2_1",
    "steepest_nodes_random", "steepest_nodes_greedy",
    "steepest_edges_random", "steepest_edges_greedy",
    "greedy_nodes_random", "greedy_nodes_greedy",
    "greedy_edges_random", "greedy_edges_greedy",
]
DATA_DIR = "../data"
RESULTS_DIR = "results"
OUT_FILE = f"{RESULTS_DIR}/checker_all.xlsx"


def load_instance(name):
    pts = []
    for line in open(f"{DATA_DIR}/{name}.csv"):
        p = line.replace(";", " ").replace(",", " ").split()
        if len(p) >= 3:
            pts.append((float(p[0]), float(p[1]), int(float(p[2]))))
    return pts


def dist(a, b):
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def evaluate(pts, sol):
    total = 0
    for i in range(len(sol)):
        a = pts[sol[i]]
        b = pts[sol[(i + 1) % len(sol)]]
        total += a[2] + dist(a, b)
    return total


def load_solution(inst, method):
    with open(f"{RESULTS_DIR}/best_{inst}_{method}.txt") as f:
        return [int(x) for x in f.read().split()]


def load_expected():
    expected = {}
    with open(f"{RESULTS_DIR}/results.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected[(row["instance"], row["method"])] = int(row["min"])
    return expected


def main():
    matches = glob.glob("../checker/*.xlsx")
    if not matches:
        print("Nie znaleziono checkera w ../checker")
        return
    template = load_workbook(matches[0])
    expected = load_expected()
    points = {inst: load_instance(inst) for inst in INSTANCES}

    summary = template.create_sheet("Summary", 0)
    headers = ["Instance", "Method", "results.csv (min)", "Python recomputed",
               "Checker K2", "Python = results.csv", "Checker = results.csv"]
    for c, h in enumerate(headers, start=1):
        cell = summary.cell(row=1, column=c, value=h)
        cell.font = Font(bold=True)
    row = 2
    all_ok = True
    for inst in INSTANCES:
        pts = points[inst]
        k = (len(pts) + 1) // 2
        for method in METHODS:
            path = f"{RESULTS_DIR}/best_{inst}_{method}.txt"
            if not os.path.exists(path):
                print("brak pliku:", path)
                continue
            sol = load_solution(inst, method)
            valid = len(sol) == k and len(set(sol)) == k and all(0 <= v < len(pts) for v in sol)
            value = evaluate(pts, sol)
            exp = expected[(inst, method)]
            ok = valid and value == exp
            all_ok = all_ok and ok
            print(f"{inst} {method:28s} python={value:7d} results.csv={exp:7d} "
                  f"valid={valid} {'OK' if ok else 'BLAD'}")

            sheet_name = f"{inst}_{method}"
            ws = template.copy_worksheet(template[inst])
            ws.title = sheet_name
            for i in range(100):
                ws.cell(row=3 + i, column=6).value = None
            for i, node in enumerate(sol):
                ws.cell(row=3 + i, column=6).value = node

            summary.cell(row=row, column=1, value=inst)
            summary.cell(row=row, column=2, value=method)
            summary.cell(row=row, column=3, value=exp)
            summary.cell(row=row, column=4, value=value)
            summary.cell(row=row, column=5, value=f"={sheet_name}!K2")
            summary.cell(row=row, column=6, value=f'=IF(D{row}=C{row},"OK","ERROR")')
            summary.cell(row=row, column=7, value=f'=IF(E{row}=C{row},"OK","ERROR")')
            row += 1

    for inst in INSTANCES:
        del template[inst]
    for col, width in zip("ABCDEFG", [10, 30, 18, 20, 14, 22, 22]):
        summary.column_dimensions[col].width = width
    template.save(OUT_FILE)
    print()
    print("PYTHON:", "WSZYSTKO OK" if all_ok else "SA BLEDY")
    print("Zapisano:", OUT_FILE)
    print("Otworz ten plik w Excelu: w arkuszu Summary kolumna 'Checker = results.csv' ma pokazac same OK.")


if __name__ == "__main__":
    main()