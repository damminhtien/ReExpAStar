# Makefile
PY ?= python3
SRC := src tests

PYLINT := $(PY) -m pylint
ISORT := $(PY) -m isort

.PHONY: lint fix imports fmt _imports_with_isort

lint:
	$(PYLINT) --rcfile=.pylintrc $(SRC)

# ========== IMPORTS ONLY ==========
# Ưu tiên isort để sắp xếp import

imports: _imports_with_isort
_imports_with_isort:
	@$(ISORT) --version >/dev/null 2>&1 && \
		$(ISORT) $(SRC) || \
		( echo ">> Install isort trước: pip install isort"; exit 1 )

# ========== FULL AUTO-FIX ==========
fix:
	$(MAKE) imports

fmt: fix
