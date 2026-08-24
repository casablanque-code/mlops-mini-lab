.PHONY: venv install data prepare train evaluate serve test lint compose-up compose-down mlflow-ui

venv:
	python3 -m venv .venv
	@echo "Activate it with: source .venv/bin/activate"

install:
	pip install -r requirements.txt

data:
	python src/data/make_dataset.py

prepare:
	python src/data/prepare.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

pipeline: data prepare train evaluate

serve:
	uvicorn src.serve.app:app --reload --port 8000

test:
	pytest tests -v

lint:
	ruff check src tests

compose-up:
	docker compose up -d mlflow serve

compose-train:
	docker compose --profile train run --rm train

compose-down:
	docker compose down

mlflow-ui:
	mlflow ui --port 5000
