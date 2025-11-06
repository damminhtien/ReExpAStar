# Contributing

Thanks for your interest!

## Dev setup

```bash
git clone https://github.com/damminhtien/reexpastar.git
cd reexpastar
pip install -e .[dev,plot]
pre-commit install
```

## Style & checks

- Formatting: `black` (line length 100) + `isort` (black profile)
- Lint: `ruff`
- Types: `mypy`
- Tests: `pytest` or `unittest` + coverage

Run all locally:

```bash
black --check . && mypy src && pytest -q
```

## Pull requests

- Include tests for new features
- Update docs and README examples when needed
- Keep commits small and focused
