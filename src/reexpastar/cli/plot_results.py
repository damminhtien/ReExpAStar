import argparse
import csv
import importlib
from typing import Any

PLT: Any | None
IMPORT_ERROR: Exception | None
try:
    PLT = importlib.import_module("matplotlib.pyplot")
except ImportError as exc:  # pragma: no cover
    PLT = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def load_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def main():
    if PLT is None:
        assert IMPORT_ERROR is not None
        raise RuntimeError("matplotlib is required to plot results") from IMPORT_ERROR
    p = argparse.ArgumentParser(description="Plot benchmark CSV: runtime vs reopens by scenario")
    p.add_argument("csv", help="CSV file from benchmark")
    args = p.parse_args()
    rows = load_rows(args.csv)
    assert PLT is not None
    scenarios = sorted(set(r["scenario"] for r in rows))
    for sc in scenarios:
        sub = [r for r in rows if r["scenario"] == sc]
        PLT.figure()
        x = [float(r["reopens"] or 0) for r in sub]
        y = [float(r["runtime_ms"] or 0) if r["runtime_ms"] != "None" else 0.0 for r in sub]
        labels = [r["algo"] for r in sub]
        assert PLT is not None  # kept narrow scope for type checkers
        PLT.scatter(x, y)
        for xi, yi, lab in zip(x, y, labels, strict=False):
            PLT.annotate(lab, (xi, yi))
        PLT.xlabel("Reopens")
        PLT.ylabel("Runtime (ms)")
        PLT.title(sc)
        out_png = args.csv.replace(".csv", f"_{sc}.png")
        PLT.savefig(out_png, bbox_inches="tight")
        print(out_png)


if __name__ == "__main__":
    main()
