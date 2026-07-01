# Graph AML on Elliptic (Phase 4)

Elliptic: 203769 nodes, 165 features, 234355 edges, timesteps 1-49.
Labeled 46564 (4545 illicit = 9.8%); the rest unknown and masked. Trained on cpu.

## Temporal split (no leak)
Train timestep <= 34, test > 34 (Weber et al. split). Train nodes 29894, test nodes 16670.
train.max(ts)=34 < test.min(ts)=35. Asserted in tests/test_graph_split.py.
Unknown nodes still pass messages but are not in the loss.

## Full features (165): tabular vs GNN
| Model | illicit F1 | precision | recall | PR-AUC |
|---|---|---|---|---|
| LightGBM (tabular) | 0.812 | 0.913 | 0.732 | 0.806 |
| GraphSAGE | 0.522 | 0.515 | 0.528 | 0.493 |
| GAT | 0.529 | 0.504 | 0.556 | 0.501 |

## Local features (94, no aggregates): tabular vs GNN
| Model | illicit F1 | precision | recall | PR-AUC |
|---|---|---|---|---|
| LightGBM (tabular) | 0.759 | 0.810 | 0.715 | 0.788 |
| GraphSAGE | 0.434 | 0.308 | 0.732 | 0.557 |

## Main conclusion (the point of this project)
On the full feature set the tabular model beats the GNN (F1 0.812 vs 0.529): ~71 features are already
neighborhood aggregates, so the graph structure is baked into the node features
(Weber et al. 2019: RandomForest illicit-F1 0.79 vs GCN 0.42).

On local features (aggregates removed) the tabular model still beats the GNN (F1 0.759 vs 0.434). The GNN trades precision
for recall (high recall, low precision) - it flags many nodes and catches most illicit,
but noisily.

Takeaway (methodological, not a failure): a GNN is not a default win. Validate it against
a strong gradient-boosted tabular baseline. On Elliptic the signal a vanilla 2-layer
GraphSAGE/GAT extracts is already captured by hand-crafted features and boosting; beating
the tabular model needs a stronger graph model (temporal GNNs like EvolveGCN, tuned
nets). Knowing when NOT to reach for a GNN is the result.

## Suspicious subgraph (compliance report)
Top illicit-scored test node 194140 (score 1.0): 10 neighbors, top neighbor scores [0.9827, 0.0039, 0.0002, 0.0, 0.0].
GNNExplainer or GAT attention weights can refine which edges drive the flag.
