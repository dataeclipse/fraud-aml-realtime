# Real-time Fraud + Graph AML

[![CI](https://github.com/dataeclipse/fraud-aml-realtime/actions/workflows/ci.yml/badge.svg)](https://github.com/dataeclipse/fraud-aml-realtime/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/deps-uv-purple)
![License](https://img.shields.io/badge/license-MIT-green)

Two connected subsystems in one repo:

- **A. Real-time fraud** - a transaction stream feeds online features, a low-latency GBDT scorer
  and a rule engine return an allow / review / block decision, with fraud-rate and drift monitoring.
- **B. Graph AML** - a GNN (GraphSAGE / GAT) over the transaction graph flags illicit nodes and
  suspicious clusters that a tabular model misses, with neighbour/subgraph explanations.

> Status: Phase 0 (skeleton). Structure, tooling, CI, and `/healthz` are in place. Models and
> business logic land in Phases 1-5 (see [Roadmap](#roadmap)).

## Problem
Banks treat fraud prevention and AML/CFT as a priority. A single classifier is not enough: the
production loop is stream -> online features -> model + rules -> decision -> monitoring, plus a
graph module that catches laundering patterns (smurfing, cycles) invisible to a tabular model.

## Data
Both real, from Kaggle. `data/` is in `.gitignore`.
- **IEEE-CIS Fraud Detection** (competition `ieee-fraud-detection`, by Vesta) - real e-commerce
  transactions, ~590k rows, ~3.5% fraud, rich anonymized features + identity. Subsystem A.
  Accept the competition rules once on Kaggle, otherwise the download fails.
- **Elliptic Data Set** (dataset `ellipticco/elliptic-data-set`) - a real Bitcoin transaction
  graph, ~203k nodes, licit/illicit labels, 49 time steps. Subsystem B (the method transfers to a
  bank graph of client <-> account <-> counterparty).

## Architecture
```mermaid
flowchart TD
    subgraph A[A. Real-time fraud]
        T[Replay producer: IEEE-CIS -> Kafka] --> W[Stream features: Bytewax windows]
        W --> FS[(Feast / Redis online store)]
        FS --> S[FastAPI /score: LightGBM + rule engine]
        S --> D[Decision: allow / review / block]
        S --> MON[Evidently + Prometheus: drift, fraud-rate, latency]
    end
    subgraph B[B. Graph AML]
        E[Elliptic -> transaction graph] --> G[GNN: GraphSAGE / GAT]
        G --> P[Illicit probability]
        P --> X[Explain: important neighbours / subgraph]
        X --> R[Report: suspicious clusters]
    end
```

## How to run
Requires [uv](https://docs.astral.sh/uv/). From `02-fraud-aml-realtime/`:
```bash
make install          # uv sync --extra data (core + dev + data layer)
make lint             # ruff check + ruff format --check
make type             # mypy strict on src
make test             # pytest
make run              # uvicorn on :8000, then curl http://localhost:8000/healthz
```
No `make` on Windows: call the `uv run ...` equivalents directly.

Heavy stacks are optional extras (installed per phase):
```bash
uv sync --extra ml        # tabular fraud + GNN (lightgbm/xgboost/shap, torch, torch-geometric)
uv sync --extra stream    # kafka/bytewax/feast/redis
```
GPU note: install `torch` (and then `torch-geometric`) from the PyTorch CUDA index for the RTX 4070;
`uv.lock` pins the CPU build from PyPI for reproducibility. Install `torch-geometric` after `torch`.

## Results
**Phase 1 - tabular fraud baseline** (IEEE-CIS, 590,540 rows x 434 columns, ~3.5% fraud, time-based
split). LightGBM with `scale_pos_weight`, honest late-period test:

| Metric | Validation | Test (late period) |
|---|---|---|
| PR-AUC | 0.394 | 0.263 |
| ROC-AUC | 0.879 | 0.845 |

The PR-AUC drop from validation to the later test period is temporal drift that a time-based split
exposes and a random split would hide. Threshold picked by expected cost (FN=10, FP=1). Details:
[docs/fraud_baseline.md](docs/fraud_baseline.md), [docs/split.md](docs/split.md),
[docs/eda.md](docs/eda.md), curve: [docs/img/pr_curve.png](docs/img/pr_curve.png).

## Roadmap
| Phase | Content |
|---|---|
| 0 ✅ | Skeleton: structure, uv/pyproject + extras, ruff/mypy/pytest/pre-commit, CI, `/healthz` |
| 1 ✅ | Tabular fraud baseline (IEEE-CIS): time-based split, LightGBM, cost threshold, MLflow (test ROC-AUC 0.845) |
| 2 ✅ | Streaming: Redpanda replay + Bytewax windows + Feast/Redis; one update_window for batch and stream (no skew) |
| 3 | Real-time service + rule engine, allow/review/block, p99 SLA, Prometheus |
| 4 | Graph AML (GNN on Elliptic), beats tabular baseline on illicit F1, subgraph explanations |
| 5 | Monitoring + compose (kafka/redis/api/prometheus), demo, model card |

## License
[MIT](LICENSE).
