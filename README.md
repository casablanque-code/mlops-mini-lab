# mlops-mini-lab

A small but complete home MLOps project: synthetic data → model training
with experiment tracking → inference over a REST API → CI/CD on GitHub
Actions → Docker Compose. Everything runs on a laptop, no GPU, no
external datasets to download.

The goal of this repo isn't the model itself (it's a toy RandomForest
on synthetic data) — it's the end-to-end plumbing around it: data
versioning, experiment tracking, reproducible pipelines, containerized
training/serving, and an automated CI/CD flow. That plumbing is the
same regardless of whether the model behind it is a toy classifier or
a real production model.

## Stack

| Concern              | Tool                                  |
|-----------------------|----------------------------------------|
| Data generation        | `scikit-learn` (`make_classification`) |
| Data/pipeline versioning | DVC                                  |
| Model training          | scikit-learn (RandomForest)           |
| Experiment tracking     | MLflow                                |
| Inference API           | FastAPI + uvicorn                     |
| Containerization        | Docker, Docker Compose                |
| CI/CD                  | GitHub Actions → GHCR                 |
| (stage 2)               | k3d/minikube + Argo CD for GitOps     |

## How the pipeline fits together

```
make_dataset.py  →  prepare.py  →  train.py  →  evaluate.py  →  serve/app.py
    (data)            (split)       (model)      (metrics)        (API)
```

1. **`src/data/make_dataset.py`** generates a synthetic binary
   classification dataset (`sklearn.datasets.make_classification`).
   No downloads, fully reproducible via `--seed`.
2. **`src/data/prepare.py`** splits the data into train/test
   (80/20, stratified by class) — stands in for a real preprocessing
   stage.
3. **`src/train.py`** trains a `RandomForestClassifier` and logs the
   run to MLflow: hyperparameters, metrics, and the model itself as
   an artifact. This is what "experiment tracking" buys you — every
   run is visible and comparable in the MLflow UI.
4. **`src/evaluate.py`** is a separate evaluation step against the
   test set; it writes `metrics.json`, which DVC picks up as a
   pipeline metric (`dvc metrics show`), so you can diff metrics
   across commits/branches.
5. **`src/serve/app.py`** is a standalone inference process: it loads
   `model.joblib` once at startup and serves `/predict`. Training and
   serving are deliberately separate — training is heavy and
   infrequent, serving is light and frequent, so they get different
   Docker images, different resource profiles, and different
   lifecycles.

`docker-compose.yml` wraps this into three services (`mlflow`,
`train`, `serve`). CI runs the full pipeline on a small sample on
every PR to catch breakage before merge. CD builds and pushes the
`train`/`serve` images to GHCR after every merge to `main`.

## Prerequisites

- Python 3.11+ (3.12 also works)
- Docker + Docker Compose v2 (`docker compose version`, no dash)
- Git

## Quick start (local, no Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# full pipeline
make pipeline
# equivalent to:
#   python src/data/make_dataset.py
#   python src/data/prepare.py
#   python src/train.py
#   python src/evaluate.py

# start the inference service locally
make serve
# in another terminal:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, -0.3, 0.4, 0.0, 0.7, -0.1, 0.3]}'
```

`/features` must have as many values as the dataset has feature
columns (8 by default — see `--n-features` in `make_dataset.py`).

## Using Docker Compose

```bash
# 1. train a model (one-off job via the "train" profile)
docker compose --profile train run --rm train

# 2. start the MLflow UI and the inference service
docker compose up -d mlflow serve

# MLflow UI:      http://localhost:5000
# Inference API:  http://localhost:8000/docs
```

## Running on Kubernetes (k3d)

Local Kubernetes deployment via k3d with a built-in registry.

```bash
# 1. create a cluster with a local registry, port 8080 -> ingress
k3d cluster create mlops-lab \
  --registry-create mlops-registry:0.0.0.0:5000 \
  --port "8080:80@loadbalancer" \
  --agents 1

# 2. build and push the serve image to the cluster registry
docker build -t localhost:5000/mlops-mini-lab-serve:v1 -f docker/Dockerfile.serve .
docker push localhost:5000/mlops-mini-lab-serve:v1

# 3. deploy via Helm
helm install mlops-serve deploy/serve-chart

# 4. test through the traefik ingress
curl -H "Host: serve.mlops.local" http://localhost:8080/health
curl -X POST -H "Host: serve.mlops.local" -H "Content-Type: application/json" \
  http://localhost:8080/predict \
  -d '{"features": [0.1, 0.2, -0.3, 0.4, 0.0, 0.7, -0.1, 0.3]}'
```

Note: images must be referenced as `mlops-registry:5000/...` (not
`localhost:5000/...`) inside Kubernetes manifests — that's the
registry's address as seen from inside the cluster network, set up
via k3d's `registries.yaml`.


## DVC pipeline (reproducibility)

```bash
dvc init          # once per repo
dvc repro          # re-runs make_dataset → prepare → train → evaluate,
                    # but only the stages whose inputs actually changed
dvc metrics show    # prints metrics.json
```

## Tests and linting

```bash
make test
make lint
```

CI runs the same two commands, plus a smoke run of the whole pipeline
on a small sample, plus a build of both Docker images.

## CI/CD

- **CI** (`.github/workflows/ci.yml`): lint, unit tests, a smoke run
  of the full pipeline on a small sample, and a build of both Docker
  images. Runs on every PR and on every push to `main`.
- **CD** (`.github/workflows/cd.yml`): on every push to `main`, builds
  and pushes the `serve` and `train` images to the GitHub Container
  Registry, tagged `latest` and `<commit-sha>`.

## Project layout

```
mlops-mini-lab/
├── data/
│   ├── raw/                # generated dataset (gitignored, DVC-tracked)
│   └── processed/          # train/test split (gitignored, DVC-tracked)
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve/
│       └── app.py
├── models/                 # trained model artifact (gitignored)
├── tests/
│   └── test_pipeline.py
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── docker-compose.yml
├── dvc.yaml                # pipeline DAG for `dvc repro`
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── requirements.txt
├── pyproject.toml          # pytest + ruff config
├── Makefile
└── README.md
```

## Roadmap (stage 2)

- [x] Deploy `serve` to k3d/minikube via a Helm chart
- [ ] Argo CD watching a `deploy/` folder — full GitOps loop
- [ ] MLflow Model Registry with staging → production promotion
- [ ] Data drift monitoring (evidently / whylogs)

## Documentation

- [docs/architecture.md](docs/architecture.md) -- what each component
  does, why it's there, and how they connect (with a diagram)
- [docs/runbook.md](docs/runbook.md) -- step-by-step commands for
  every workflow, plus a troubleshooting section
- [docs/decisions.md](docs/decisions.md) -- why things are built the
  way they are, alternatives considered
