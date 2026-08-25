# Runbook

Practical, step-by-step instructions for running every part of this
project and troubleshooting the common failure modes. For the "why",
see [architecture.md](architecture.md).

## Prerequisites

- Python 3.11+ (3.12 also works)
- Docker + Docker Compose v2 (`docker compose version` -- no dash)
- Git

## venv cheat sheet

```bash
python3 -m venv .venv          # create, once per project
source .venv/bin/activate      # activate -- do this at the start of every session
deactivate                     # exit, from anywhere

which python                    # sanity check: should point into .venv/bin/python
```

To reset a broken environment:
```bash
deactivate 2>/dev/null
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`git` commands work the same whether the venv is active or not --
`.venv/` is gitignored and never touched by Git.

## Local setup (no Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify:
```bash
which python                    # should be .../mlops-mini-lab/.venv/bin/python
pip list | grep -E "scikit-learn|mlflow|dvc|fastapi"
```

## Running the pipeline

```bash
make pipeline
# equivalent to:
#   python src/data/make_dataset.py
#   python src/data/prepare.py
#   python src/train.py
#   python src/evaluate.py
```

Expected output ends with something like:
```
Metrics: {'accuracy': 0.9, 'f1': 0.88, 'roc_auc': 0.95}
```

Check the artifacts landed where expected:
```bash
ls data/raw/dataset.csv data/processed/*.csv models/model.joblib metrics.json
```

## Running tests and lint

```bash
make test     # pytest tests -v
make lint     # ruff check src tests
```

This is exactly what CI runs -- if it's green locally, CI should be
green too (see [Troubleshooting](#troubleshooting) if it isn't).

## Serving locally

```bash
make serve
# in another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, -0.3, 0.4, 0.0, 0.7, -0.1, 0.3]}'
```

`features` must have exactly `N_FEATURES` values (8 by default, set
via the `N_FEATURES` env var and matching `--n-features` used when the
dataset was generated). Swagger UI is available at
`http://localhost:8000/docs`, and Prometheus metrics at
`http://localhost:8000/metrics`.

## MLflow UI

```bash
make mlflow-ui
```
Open `http://localhost:5000` to see logged runs, parameters, and
metrics.

## Registering a model and promoting it to Production

By default `train.py` only saves a local `.joblib` file. To register a
version in the MLflow Model Registry:

```bash
python src/train.py --registered-model-name mlops-mini-lab
```

Then set an alias on a version (check the version number in the
MLflow UI under Models, or via `mlflow models
get-model-version-by-alias`). **Note:** MLflow deprecated stages
(`Staging`/`Production`) in favor of aliases -- use `set-registered-model-alias`,
not `transition-stage`:

```bash
mlflow models set-registered-model-alias \
  --name mlops-mini-lab --alias production --version 1
```

Point the serving service at the registry instead of the local file,
using the `@alias` URI syntax:
```bash
export MODEL_URI=models:/mlops-mini-lab@production
export MLFLOW_TRACKING_URI=http://localhost:5000
make serve
```

To roll back, just repoint the same alias at a previous version and
restart the serving process (the model is cached in memory, so a
running process won't pick up the change until it reloads):
```bash
mlflow models set-registered-model-alias \
  --name mlops-mini-lab --alias production --version 1
```

**Note:** the Model Registry requires a database-backed MLflow
tracking server (not the default `file:./mlruns`). Start one with:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

## Docker Compose

```bash
# one-off training job
docker compose --profile train run --rm train

# long-running services
docker compose up -d mlflow serve
curl http://localhost:8000/health

# tear down
docker compose down
```

## DVC pipeline

```bash
dvc init            # once per repo
dvc repro            # re-runs only the stages whose inputs changed
dvc metrics show      # prints metrics.json
```

On a second `dvc repro` with nothing changed, expect:
```
Stage 'evaluate' didn't change, skipping
```
That's the point -- it only recomputes what's actually stale.

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'` when running pytest**
Make sure `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```
This tells pytest to add the repo root to `sys.path` so `from
src.serve.app import app` resolves.

**`sklearn` warns "X does not have valid feature names"**
The model was trained on a `DataFrame` with column names, but
`predict()`/`predict_proba()` was called with a plain list or an
unnamed `DataFrame`. `serve/app.py` already wraps inference input in a
`DataFrame` using `model.feature_names_in_` to avoid this -- if you
see the warning elsewhere, check that any new inference code does the
same.

**`ValueError: Out of range float values are not JSON compliant`
when testing invalid input (e.g. NaN)**
Starlette's `JSONResponse` calls `json.dumps(..., allow_nan=False)`.
If a Pydantic validator rejects a `NaN`/`Inf` value, FastAPI's default
error handler still tries to echo that raw value back in the `422`
response body, which then fails to serialize. Fixed by a custom
`RequestValidationError` handler in `serve/app.py` that strips
`NaN`/`Inf` from the error payload before returning it. If this
resurfaces elsewhere, look for any endpoint returning user-supplied
floats verbatim in an error response.

**`docker-compose` (with a dash) behaves oddly / unsupported syntax**
That's the old Python-based standalone binary (v1, unmaintained).
Check `docker compose version` (no dash) -- Docker Engine 20.10+
ships the v2 plugin. Use the no-dash form everywhere in this repo.

**CI green locally but red on GitHub Actions (or vice versa)**
Usually a Python version mismatch. CI pins `python-version: "3.11"` in
`.github/workflows/ci.yml`; if your local venv is on a different minor
version, dependency resolution can differ subtly. Match the CI version
locally when debugging a CI-only failure.

**`mlflow models transition-stage` doesn't exist / stages don't appear
in the UI**
MLflow deprecated stages (`Staging`/`Production`) in favor of aliases
starting around 2.9. Use `mlflow models set-registered-model-alias`
and the `models:/<name>@<alias>` URI syntax instead -- see
[Registering a model](#registering-a-model-and-promoting-it-to-production)
above.

**`dvc repro` works but there's no `dvc.lock` in the repo**
`dvc.lock` is generated the first time `dvc repro` actually runs (it
records the exact hashes of inputs/outputs for each stage) and should
be committed alongside `dvc.yaml` -- it's what lets a teammate (or CI)
verify the pipeline reproduces the same result, and what `dvc repro`
compares against to decide whether a stage needs to re-run at all. If
you don't see it, run `dvc init` (once) and `dvc repro`, then `git add
dvc.lock`.
