# Model Card - Real-time Fraud + Graph AML

Model card in the spirit of bank model validation and AML/CFT. Two subsystems, versioned with the code.

| Field | Value |
|---|---|
| System | (A) real-time fraud scoring, (B) graph AML |
| Code version | fraud-aml 1.0.0 |
| Fraud model | LightGBM `phase1-lgbm` (baked to deploy/) |
| AML models | LightGBM tabular baseline, GraphSAGE, GAT (torch-geometric) |
| Date | 2026-07-01 |
| Owner / validator | dataeclipse (DS) / independent validation required before production |
| Status | Release v1.0.0 (demo/portfolio, not a live lending or AML decision) |

## 1. Purpose and scope
- **A. Real-time fraud** - score a transaction at authorization time for a allow / review / block
  decision (decision support, not a silent auto-block).
- **B. Graph AML** - classify graph nodes as licit/illicit and surface suspicious subgraphs for a
  compliance analyst.
- **Out of scope:** silent automated blocking without human review; use as the sole basis of an
  AML report; transfer to another population or graph without revalidation.

## 2. Data
- **IEEE-CIS Fraud Detection** (Kaggle, by Vesta) - real e-commerce transactions, 590,540 rows x
  434 columns, ~3.5% fraud (~1:28). Subsystem A. Anonymized features (proxy domains).
- **Elliptic Data Set** - real Bitcoin transaction graph, 203,769 nodes, 165 features, 234,355
  edges, 49 time steps; 46,564 labeled (9.8% illicit). Subsystem B.
- **Limitations:** both datasets are real but anonymized; the exact features are proxies. The AML
  method transfers to a bank graph (client <-> account <-> counterparty), but node features and
  label definitions would be bank-specific and need re-derivation and revalidation.

## 3. Methodology
- **A**: strict time-based split (train early, test late); leak-safe frequency/target encoding fit
  on train only; the same `assemble_features` builds the vector at train and serve time (no skew);
  LightGBM with `scale_pos_weight`; a cost-optimal threshold from FN/FP prices.
- **B**: strict temporal split by time step (train <= 34, test > 34, Weber et al.); a LightGBM
  tabular baseline and GraphSAGE/GAT GNNs on the full 165 features and on the local 94 (aggregates
  removed); illicit-class metrics, not accuracy.

## 4. Metrics
**A. Fraud** (holdout by time; validation vs late-period test):

| Metric | Validation | Test (late period) |
|---|---|---|
| PR-AUC | 0.394 | 0.263 |
| ROC-AUC | 0.879 | 0.845 |

Decision threshold picked by expected cost (FN=10, FP=1): 0.0948. The PR-AUC drop is temporal drift
the time split exposes.

**B. AML** (illicit F1 on the temporal test set):

| Feature set | LightGBM | GraphSAGE | GAT |
|---|---|---|---|
| Full (165) | **0.812** | 0.522 | 0.529 |
| Local (94) | **0.759** | 0.434 | - |

The boosted tabular model beats the vanilla GNN on both sets. See section 9.

## 5. No train/serve skew
A single `assemble_features` builds the fraud feature vector for training (export) and for serving;
`tests/test_no_skew_serving.py` asserts they match for the same transaction. Streaming window
features use a single `update_window` for the offline applier and the online Bytewax operator, with
an equivalence test over edge cases (ties, window boundary, sparse gaps, unseen entity).

## 6. Decision logic
The ML score maps to bands (block/review/allow) from the cost threshold. The velocity rule engine
(count / velocity / window-sum over online features) only ESCALATES strictness (forces review or
block), never lowers the ML decision.

## 7. Latency
Local `/score` p50 56.5 ms / p99 70.9 ms over 500 transactions. Component budget: feature-build
43 ms (the bottleneck), model-predict 2.6 ms, pred_contrib reason 8.5 ms, rules and serialize
<0.1 ms. Reason codes use native LightGBM `pred_contrib`, not SHAP per request. **Honest note:**
these exclude the Feast/Redis online lookup (a network round-trip, ~0.2-1 ms local Redis, more if
remote); the local p99 is not the full production p99.

## 8. Explainability
- **A**: per-transaction top reason from native `pred_contrib`, plus the fired rules.
- **B**: a k-hop suspicious subgraph for a flagged node (neighbors and their scores) as a compliance
  report; GNNExplainer or GAT attention can refine which edges drive the flag.

## 9. Limitations and risks
- **Temporal drift (fraud):** the time split shows a real PR-AUC drop from validation to the later
  test period; the model degrades over time and must be revalidated and refreshed.
- **GNN does not beat the tabular baseline (AML), and why:** on Elliptic ~71 of the 165 features are
  already neighborhood aggregates, so the graph signal is baked into the node features (Weber et al.
  2019: RandomForest illicit-F1 0.79 vs GCN 0.42). Even on local features the boosted tabular model
  wins on F1; the GNN reaches high recall at low precision. The lesson is methodological - a GNN is
  not a default win; beating a strong boosted baseline needs a stronger graph model (temporal GNNs
  like EvolveGCN, tuning).
- **Cost threshold depends on FN/FP prices:** the block/review bands move with the assumed cost of a
  missed fraud vs a false alarm; these must be set with the business.
- **Anonymized proxies; no reject inference:** external validity is limited; the model learned on
  labeled/known transactions only.

## 10. Monitoring requirements
- **Fraud rate and input drift** over time (PSI/Evidently); alert on a rising fraud-rate or PSI.
- **Latency SLA** (p99) and the decision mix (allow/review/block rate) via Prometheus.
- **Revalidation** at least every 6-12 months or on a drift alert or a drop in fraud metrics on a
  fresh labeled slice. AML thresholds and rules reviewed with compliance.

## 11. Governance and versioning
Reproducible: fixed seeds, `uv.lock`, baked model (`fraud-aml-export-model` -> deploy/). Before any
production use: independent model validation (back-testing, stability, fairness/AML review), a risk
and compliance sign-off, and an audit trail of versions and decisions. Current status is
demo/portfolio.
