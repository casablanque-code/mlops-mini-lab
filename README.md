# mlops-mini-lab

Маленький, но полный домашний MLOps-проект: синтетические данные →
обучение с трекингом в MLflow → инференс через FastAPI → CI/CD на
GitHub Actions → Docker Compose. Всё крутится на ноуте, без GPU и
внешних датасетов.

## Стек

- **Данные**: синтетика (`sklearn.make_classification`), версии — DVC
- **Обучение**: scikit-learn (RandomForest), трекинг — MLflow
- **Инференс**: FastAPI + uvicorn
- **Контейнеризация**: Docker, Docker Compose
- **CI/CD**: GitHub Actions → GHCR
- **(этап 2)**: k3d/minikube + Argo CD для GitOps-деплоя

## Быстрый старт (локально, без Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# полный пайплайн
make pipeline
# эквивалентно:
#   python src/data/make_dataset.py
#   python src/data/prepare.py
#   python src/train.py
#   python src/evaluate.py

# поднять инференс локально
make serve
# в другом терминале:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, -0.3, 0.4, 0.0, 0.7, -0.1, 0.3]}'
```

## Через Docker Compose

```bash
# 1. обучить модель (одноразовый job-профиль)
docker compose --profile train run --rm train

# 2. поднять mlflow UI и инференс-сервис
docker compose up -d mlflow serve

# MLflow UI: http://localhost:5000
# Inference API: http://localhost:8000/docs
```

## DVC-пайплайн (воспроизводимость)

```bash
dvc init          # один раз
dvc repro          # прогонит make_dataset → prepare → train → evaluate
dvc metrics show    # покажет metrics.json
```

## Тесты и линт

```bash
make test
make lint
```

## CI/CD

- **CI** (`.github/workflows/ci.yml`): линт, юнит-тесты, smoke-прогон
  всего пайплайна на маленькой выборке, сборка обоих Docker-образов —
  запускается на каждый PR и push в `main`.
- **CD** (`.github/workflows/cd.yml`): при мерже в `main` собирает и
  пушит `serve`/`train` образы в GitHub Container Registry с тегами
  `latest` и `<sha>`.

## Roadmap (этап 2)

- [ ] Развернуть `serve` в k3d/minikube через Helm-чарт
- [ ] Argo CD, следящий за папкой `deploy/` — полноценный GitOps
- [ ] Model Registry в MLflow с promotion staging → production
- [ ] Мониторинг дрейфа данных (evidently / whylogs)
