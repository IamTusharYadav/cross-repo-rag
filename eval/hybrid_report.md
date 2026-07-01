# Retrieval Layer Ablation Benchmark Report
Generated on: 2026-07-01

## Aggregated Pipeline Performance Comparison
| Metric           |   Hybrid (Dense + BM25 + RRF) Baseline |   Hybrid + AST Rerank |   Hybrid + Repo Expansion + AST |
|:-----------------|---------------------------------------:|----------------------:|--------------------------------:|
| MRR              |                               0.220561 |              0.281053 |                        0.335204 |
| Recall@1         |                               0.133333 |              0.216667 |                        0.233333 |
| Recall@3         |                               0.216667 |              0.216667 |                        0.25     |
| Recall@5         |                               0.25     |              0.283333 |                        0.333333 |
| Recall@10        |                               0.333333 |              0.45     |                        0.5      |
| P95 Latency (ms) |                             164.465    |            119.154    |                      220.954    |
