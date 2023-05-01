import argparse, csv
import matplotlib.pyplot as plt


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description="Plot benchmark CSV: runtime vs reopens by scenario")
    p.add_argument("csv", help="CSV file from benchmark")
    args = p.parse_args()
    rows = load_rows(args.csv)
    scenarios = sorted(set(r["scenario"] for r in rows))
    for sc in scenarios:
        sub = [r for r in rows if r["scenario"] == sc]
        plt.figure()
        x = [float(r["reopens"] or 0) for r in sub]
        y = [float(r["runtime_ms"] or 0) if r["runtime_ms"] != "None" else 0.0 for r in sub]
        labels = [r["algo"] for r in sub]
        plt.scatter(x, y)
        for xi, yi, lab in zip(x, y, labels):
            plt.annotate(lab, (xi, yi))
        plt.xlabel("Reopens")
        plt.ylabel("Runtime (ms)")
        plt.title(sc)
        out_png = args.csv.replace(".csv", f"_{sc}.png")
        plt.savefig(out_png, bbox_inches="tight")
        print(out_png)


if __name__ == "__main__":
    main()
