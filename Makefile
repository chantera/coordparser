PYTHON := python3

.PHONY: all
all: build

.PHONY: build
build:
	@$(PYTHON) setup.py build_ext --inplace

.PHONY: clean
clean:
	@rm -rf build src/parsers/cky.cpp src/parsers/*.so
