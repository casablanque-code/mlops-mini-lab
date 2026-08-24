# Design decisions

Short ADR-style entries: what was decided, what alternatives existed,
why this option won. Newest at the bottom.

---

## 1. Synthetic data instead of a real dataset

**Decision:** generate data with `sklearn.datasets.make_classification`
rather than downloading a real dataset (UCI, Kaggle, etc.).

**Alternatives considered:** Titanic/Wine Quality (small, well-known,
free license); a real domain dataset for portfolio weight.

**Why synthetic won:** zero external downloads means the pipeline is
fully reproducible offline and in CI with no network dependency, no
licensing question, and no risk of a dataset disappearing or changing.
The project's value is the MLOps plumbing, not the dataset -- a real
dataset can be swapped in later (`make_dataset.py` is the only file
that would need to change) without touching anything downstream.

---

## 2. Separate Dockerfiles for training and serving

**Decision:** `docker/Dockerfile.train` and `docker/Dockerfile.serve`
instead of one shared image with both entry points.

**Alternatives considered:** a single image with an entrypoint script
choosing train vs. serve mode based on an env var or CLI arg.

**Why separate won:** training and serving have different resource
profiles (training: CPU-heavy, runs once per job; serving: I/O-bound,
runs continuously) and different dependency needs in a real project
(training might need heavier libraries that serving never touches).
Keeping them separate keeps the serving image lean and makes the
distinction between "batch job" and "long-running service" explicit
in the repo structure, not just in how the image happens to be
invoked.

---

## 3. Two separate GitHub Actions workflows (CI vs CD)

**Decision:** `ci.yml` triggers on PRs and pushes to `main`; `cd.yml`
triggers only on pushes to `main`.

**Alternatives considered:** one workflow with conditional jobs based
on the event type.

**Why separate won:** the two workflows answer different questions
("is this change safe to merge" vs. "ship this to the registry") and
have different blast radii -- CD pushes artifacts to a registry that
other systems might pull from, CI does not. Keeping them as separate
files makes the trigger conditions immediately visible without reading
job-level `if:` conditions, and means CD can be re-run independently
(e.g. to re-push an image) without re-running the full test suite.

---

## 4. Image tags: both `latest` and `<commit-sha>`

**Decision:** CD pushes every image with two tags.

**Alternatives considered:** `latest` only; semantic version tags via
manual release process.

**Why both won:** `latest` is convenient for local development and
quick manual testing (`docker pull ...:latest`), but it's a moving
target and unsuitable for anything that needs to be reproducible or
auditable. The SHA tag is immutable and directly traceable to a
commit -- given a running container, you can always answer "what code
produced this" with `git show <sha>`. Semantic versioning was skipped
for now since it implies a release process (changelogs, version
bumps) that's out of scope for a lab project; the SHA tag already
gives full traceability, and semver can be layered on top later
without changing the underlying mechanism.

---

## 5. MLflow Model Registry accessed via `MODEL_URI`, defaulting to a local file

**Decision:** the serving service reads a `MODEL_URI` env var that can
be `file:./models/model.joblib`, `runs:/<id>/model`, or
`models:/<name>/<stage>`, defaulting to the local file scheme.

**Alternatives considered:** always require the Model Registry (no
local-file fallback); a separate serving code path per source.

**Why this won:** defaulting to `file:` keeps `docker compose up
serve` working immediately after `docker compose --profile train run
train`, with zero MLflow server setup required -- important for a
"just try it" experience on a laptop. Routing everything through
`mlflow.pyfunc.load_model()` for the non-file case means the serving
code doesn't need to know or care whether the model came from the
Registry, a specific run, or cloud storage -- that's MLflow's job, not
the service's.

---

## 6. Pydantic strict validation (`conlist` + custom NaN/Inf check) instead of manual checks in the endpoint

**Decision:** validation lives in the `PredictRequest` model via
`conlist(float, min_length=N, max_length=N)` and a `field_validator`,
not as `if` statements inside the `predict()` function.

**Alternatives considered:** manual length/type checks at the top of
the endpoint function.

**Why the Pydantic approach won:** invalid requests never reach
business logic at all -- FastAPI rejects them with a `422` before
`predict()` runs, and the exact reason is visible to the caller and in
`/docs` without any code reading the endpoint body. This is more
robust against forgetting a check in one code path and cheaper to
extend (adding a new constraint is one line on the model, not a new
`if` block).

**Follow-up bug this decision surfaced:** Pydantic's validation error
includes the raw invalid input in its response (`ctx.input`), and when
that input is `NaN`/`Inf`, Starlette's default JSON serialization
(`allow_nan=False`) fails while trying to return the *error itself*.
Fixed with a custom `RequestValidationError` handler that sanitizes
`NaN`/`Inf` to `null` before serializing. See
[runbook.md](runbook.md#troubleshooting) for the concrete symptom.

---

## 7. `prometheus-fastapi-instrumentator` instead of hand-rolled metrics middleware

**Decision:** use the `prometheus-fastapi-instrumentator` package to
expose `/metrics`.

**Alternatives considered:** a custom Starlette middleware
incrementing counters/histograms manually.

**Why the library won:** request counting, latency histograms, and
per-route/per-status labeling are a solved, well-tested problem;
writing it by hand risks subtly wrong bucket boundaries or missing
edge cases (streaming responses, exceptions before a response is
generated). The library also uses Prometheus's own conventions for
metric names, so a standard Prometheus/Grafana setup recognizes the
output with zero custom configuration.

---

## 8. Full Grafana/Prometheus stack deferred to stage 2 (Kubernetes)

**Decision:** `docker-compose.yml` exposes `/metrics` but does not run
Prometheus or Grafana as services.

**Alternatives considered:** add `prometheus` and `grafana` services to
`docker-compose.yml` now.

**Why deferred:** a full monitoring stack is a meaningful chunk of
infrastructure (two more services, scrape config, dashboard
provisioning) that's better introduced once the project moves to
Kubernetes, where it can be done properly with a Prometheus Operator
and a GitOps-managed config -- rather than built twice (once in
Compose, once in k8s). `/metrics` being exposed now means nothing
downstream needs to change when that stack is added later.
