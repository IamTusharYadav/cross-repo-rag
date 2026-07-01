# Retrieval Layer Ablation Benchmark Report
Generated on: 2026-06-25

## Aggregated Pipeline Performance Comparison
| Metric           |   Dense Baseline |   Dense + AST Rerank |   Dense + Repo Expansion + AST |
|:-----------------|-----------------:|---------------------:|-------------------------------:|
| MRR              |         0.199183 |             0.270185 |                       0.281296 |
| Recall@1         |         0.15     |             0.233333 |                       0.25     |
| Recall@3         |         0.216667 |             0.233333 |                       0.266667 |
| Recall@5         |         0.233333 |             0.233333 |                       0.266667 |
| Recall@10        |         0.233333 |             0.233333 |                       0.266667 |
| P95 Latency (ms) |       115.991    |            82.8883   |                     206.062    |
