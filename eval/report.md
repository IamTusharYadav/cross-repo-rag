# Retrieval Layer Ablation Benchmark Report
Generated on: 2026-07-01

## Aggregated Pipeline Performance Comparison
| Metric           |   Dense Baseline |   Hybrid (Dense + BM25 + RRF) |   Hybrid + AST |   Hybrid + Repo Expansion + Cross-Encoder |
|:-----------------|-----------------:|------------------------------:|---------------:|------------------------------------------:|
| MRR              |         0.277516 |                      0.239385 |       0.281053 |                                 0.184721  |
| Recall@1         |         0.216667 |                      0.15     |       0.216667 |                                 0.0666667 |
| Recall@3         |         0.316667 |                      0.216667 |       0.216667 |                                 0.183333  |
| Recall@5         |         0.316667 |                      0.283333 |       0.283333 |                                 0.216667  |
| Recall@10        |         0.366667 |                      0.333333 |       0.45     |                                 0.483333  |
| P95 Latency (ms) |       112.984    |                    120.311    |     100.308    |                             12361.7       |
