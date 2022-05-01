
# ReExpAStar - Weighted A* with Conditional Re-expansion (CR), NR, AR & ARA*

[![CI](https://github.com/damminhtien/reexpastar/actions/workflows/ci.yml/badge.svg)](https://github.com/damminhtien/reexpastar/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](#)
[![Lint: Ruff](https://img.shields.io/badge/lint-ruff-46b2a1.svg)](#)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](#)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-4F8EF7.svg)](#)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](#)
[![Award](https://img.shields.io/badge/Award-Best%20Paper%20(KICS%20ICTC%202023)-yellow)](#papers--recognition)

A clean, research-ready package for comparing **Weighted A\*** variants:
- **CR**: Weighted A\* with **Conditional Re-expansion** (threshold `r`, multiple modes).
- **NR**: Never Re-open.
- **AR**: Always Re-open.
- **ARA\***: Anytime Repairing A\* (baseline with reuse).

Includes **diverse scenarios** (grids, inconsistent heuristics, geometric graphs, 8-puzzle, terrain, mazes), a **benchmark suite**, and optional **plotting**.

## Install
```bash
pip install -e .[dev,plot]
pre-commit install
```

## Quickstart
```bash
# Compare on a wall+gap grid (CSV to stdout)
reexpastar-compare --width 60 --height 60 --weight 1.5 --r_cr 0.2

# Ablation of r (NR/AR/CR)
reexpastar-ablate --width 80 --height 80 --weight 1.5 --r_values "inf,1.0,0.5,0.2,0.0" --r_mode rel_edge

# Full benchmark suite (writes CSV -> results/)
reexpastar-benchmark --weight 1.5 --r_values "inf,0.5,0.2,0.0" --r_mode abs --seed 42 --include_ara
# Then produce annotated scatter plots for each scenario
reexpastar-plot results/benchmark_YYYYMMDD_HHMMSS.csv
```

## Tests
```bash
PYTHONPATH=. python -m unittest discover -s tests -t .
# or with pytest if you prefer
# pytest -q --cov=reexpastar
```

## Project layout
```
reexpastar/
  src/reexpastar/        # package code (CR/NR/AR/ARA*, grids, terrain, scenarios)
  tests/                 # unit tests
  examples/              # runnable samples
  docs/                  # MkDocs (optional)
  .github/workflows/     # CI config
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md). By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Citation
See [CITATION.cff](CITATION.cff).


## Papers & Recognition
- **Best Paper**, KICS – ICTC 2023 (selected from ~600 papers). This repository accompanies my research on re-expansion strategies in heuristic search.


### Advanced
- **Generics & Protocols** for type-safe state definitions (`reexpastar.core.types`).
- **Tie-breaking**: `tie_break='g_low'|'g_high'` to control OPEN ordering when f ties.
- **Guards & telemetry**: `max_expansions`, `max_runtime_ms`, `log_every` with `reexpastar.logging.get_logger()` (JSON-capable).
- **Parallel benchmarks**: `reexpastar.cli.benchmark_parallel` to run multiple seeds in parallel.
- **Profiling**: `reexpastar.cli.profile_search` to get cProfile snapshots on heavy instances.
