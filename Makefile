.PHONY: help env update clean

help:
	@echo "Targets:"
	@echo "  make env      Create conda env from environment.yml"
	@echo "  make update   Update env after editing environment.yml"
	@echo "  make clean    Remove the env"

env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate aidev-extended"

update:
	conda env update -f environment.yml --prune

clean:
	conda env remove -n aidev-extended