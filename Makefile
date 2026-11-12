.PHONY: install lint format typecheck test check prepare sweep compare docker

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy

test:
	pytest --cov=confusion_bert --cov-report=term-missing

check: lint typecheck test

# Requires data/raw and a label decoder (see data/README.md)
prepare:
	confusion-bert prepare --config configs/base.yaml

sweep:
	confusion-bert sweep --config configs/base.yaml --experiments configs/experiments.yaml

compare:
	confusion-bert compare --config configs/base.yaml --runs runs/bert-base-uncased_2 runs/bert-base-uncased_6 \
	  runs/bert-base-uncased_12 runs/bert-base-uncased_14 runs/bert-base-uncased_18 --metric macro_f1

docker:
	docker build -t confusion-bert .
