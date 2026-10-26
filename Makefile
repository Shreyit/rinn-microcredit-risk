# Rinn — one-command reproducibility
PY      := .venv/bin/python
TEX     := /Library/TeX/texbin
REPORT  := report/written

.PHONY: env test notebooks report clean all

env:            ## create .venv (Python 3.11) with project + dev deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

test:           ## run unit tests
	$(PY) -m pytest -q

notebooks:      ## execute notebooks top-to-bottom, in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

report:         ## compile the written report → report/written/build/main.pdf
	cd $(REPORT) && mkdir -p build && \
	PATH=$(TEX):$$PATH pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null && \
	PATH=$(TEX):$$PATH pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null && \
	PATH=$(TEX):$$PATH pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null
	@echo "→ $(REPORT)/build/main.pdf"

clean:
	rm -rf $(REPORT)/build .pytest_cache **/__pycache__

all: env test notebooks report
