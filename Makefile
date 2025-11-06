# Makefile
PY ?= python3
SRC := src tests

RUFF := $(PY) -m ruff
PYLINT := $(PY) -m pylint
ISORT := $(PY) -m isort

.PHONY: lint fix imports fmt _imports_with_ruff _imports_with_isort

lint:
	$(RUFF) check $(SRC)
	$(PYLINT) --rcfile=.pylintrc $(SRC)

# ========== IMPORTS ONLY ==========
# Ưu tiên Ruff; nếu chưa có thì fallback sang isort
imports: _imports_with_ruff

_imports_with_ruff:
	@$(RUFF) --version >/dev/null 2>&1 && \
		$(RUFF) check $(SRC) --select I --fix || \
		$(MAKE) _imports_with_isort

_imports_with_isort:
	@$(ISORT) --version >/dev/null 2>&1 && \
		$(ISORT) $(SRC) || \
		( echo ">> Install ruff hoặc isort trước: pip install ruff [hoặc] pip install isort"; exit 1 )

# ========== FULL AUTO-FIX ==========
fix:
	$(RUFF) check $(SRC) --fix
	$(RUFF) format $(SRC)

fmt: fix
