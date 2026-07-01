# Architecture Decisions Log
This document records the architectural decisions, trade-offs, and evidence behind the ingestion, retrieval, and evaluation layers of the Cross-Repo RAG system.

---

## 1. Local Cache Invalidation Topology: Relational State Ledger via SQLite

* **Context**: Parsing and embedding large repositories (`fastapi`, `qdrant-client`) on every run introduces unnecessary compute cost and vector collection drift.
* **Options Considered**:
  1. **Stateless Full Re-Indexing** — simple, but doesn't scale.
  2. **In-Vector Payload Inspections** — checking Qdrant at runtime for existing paths; expensive network round-trips, can't reliably detect in-place content edits.
  3. **Out-of-Band State Tracking Ledger (SQLite)** — track file state locally in a relational schema.
* **Decision**: Option 3.
* **Justification**: Implemented via `src.db.state_manager.IngestionStateManager`. The `needs_processing()` method gives O(1) local signature lookups, bounding embedding work strictly to new or modified files.

---

## 2. Vector Key Strategy: Deterministic Namespace-Driven UUID5

* **Context**: Vector stores need unique point IDs. Random IDs (UUID4) turn re-runs into permanent duplicate insertions instead of updates.
* **Options Considered**:
  1. **Auto-incrementing integer sequences** — breaks under multi-process writes across repos.
  2. **Random UUID4** — bloats the index on every re-run.
  3. **Deterministic UUID5** derived from a stable chunk identity string.
* **Decision**: Option 3.
* **Justification**: `uuid.uuid5(uuid.NAMESPACE_DNS, semantic_chunk_id)`. Re-running ingestion updates the same point slot instead of duplicating it — this is what makes the SQLite ledger in Decision 1 actually safe to combine with re-indexing.

---

## 3. Chunk Tracking Integrity: Structural Signature vs. Stable Semantic Address

* **Context**: Line numbers shift as code changes; a purely line-based or hash-based ID cascades into broken identities across a file after any edit.
* **Decision**: Track both identifiers as parallel metadata:
  * **`chunk_id`** (SHA-256 of literal text) — detects whether content has physically changed.
  * **`semantic_chunk_id`** (`{repo}::{symbol_path}::chunk{idx}`) — a stable logical coordinate independent of line drift, used as the UUID5 source for Decision 2.

---

## 4. Operational Resilience: Post-File Drift Pruning & Atomic State Processing

* **Context**: Edited or shrunk files leave orphaned vectors behind if old chunks aren't cleaned up; deleted files leave orphaned vectors indefinitely.
* **Design**:
  * **Atomicity**: `src/ingest.py` follows chunk → redact → embed → upsert → prune, in that order. The file is only marked processed in SQLite after every stage succeeds, so a failure mid-pipeline is retried cleanly on the next run rather than leaving a half-written state.
  * **Drift pruning**: `VectorStoreManager.prune_file_chunks()` deletes any vector tied to a file but absent from the current run's active-ID set.
  * **Orphan reconciliation**: a separate pass compares the SQLite ledger against the filesystem and removes vectors/records for files that no longer exist on disk.

---

## 5. Benchmark Redesign: From Regression Questions to a Representative Developer Benchmark

* **Context**: The original evaluation set was a small number of manually written regression-style questions — useful for smoke-testing, but too small to support statistically meaningful comparisons between retrieval configurations.
* **Decision**: Replace it with a 30-question benchmark, split across three categories — FastAPI-only, Qdrant-only, and cross-repository integration — phrased as natural developer questions rather than filename lookups, so the benchmark exercises semantic retrieval rather than string matching.
* **Consequence**: this became the fixed evaluation set for every retrieval-layer ablation described below (Decisions 6, 8, 9). Any future reranker or retrieval tuning should be checked against a held-out split of this set, not the full set, to avoid tuning decisions against the same questions used to measure them.

---

## 6. Evidence-Driven Bottleneck Diagnosis: Candidate Generation Before Reranking

* **Context**: With AST-aware reranking in place, Recall@10 on the 30-question benchmark still sat below 30% (see *Benchmark A* in Benchmark Results below) — i.e., relevant files were frequently missing from the candidate pool entirely, not merely mis-ranked within it.
* **Options Considered**:
  1. Keep tuning reranker heuristics.
  2. Add filename-specific boosting for known targets (see Decision 7 — rejected).
  3. Run a systematic evaluation to isolate whether the bottleneck was candidate generation or reranking.
* **Decision**: Option 3.
* **Finding**: correct files were frequently absent from the top-10 candidates before reranking ever ran (confirmed via deep-pool debugging in `eval_true_rank.py`, which found target files absent up to a search depth of 150). No amount of reranker tuning can recover a document that was never retrieved. This redirected effort toward candidate generation — hybrid dense + sparse retrieval (Decision 8) and repository-aware expansion (Decision 9).

---

## 7. Rejected Alternative: Filename-Specific Boosting

* **Context**: Following the diagnosis in Decision 6, one quick fix would have been to hardcode score boosts for filenames known to be benchmark targets (e.g. `qdrant_client.py`).
* **Decision**: Rejected.
* **Justification**: this would inflate benchmark metrics without improving the underlying retrieval model, and would couple the system to the naming conventions of these two specific repositories — breaking generalization to any other codebase pair a user might index.

---

## 8. Hybrid Candidate Generation: Dense + Sparse (BM25) + RRF

* **Context**: Dense retrieval alone favored explanatory documentation over implementation code, and missed files identifiable mainly by exact lexical signals — API names, filenames, class names — that don't embed distinctively.
* **Options Considered**:
  1. Dense-only (status quo).
  2. Increase dense `top_k` depth.
  3. Add a parallel sparse (BM25) lexical pathway, merged with dense via Reciprocal Rank Fusion.
* **Decision**: Option 3.
* **Implementation**: dense (BGE embeddings) and sparse (BM25) queries run in parallel; results are merged via Qdrant's native RRF.
* **Result**: see *Benchmark B* in Benchmark Results below (Hybrid baseline column) — improves candidate diversity and lexical matching for identifiers/filenames over the dense-only baseline in *Benchmark A*, without changing downstream reranking logic.

---

## 9. Repository-Aware Candidate Expansion

* **Context**: Cross-repository questions need evidence from both repos, but a single global hybrid query tends to concentrate candidates in whichever repository scores higher on average, starving the other.
* **Options Considered**:
  1. Increase global retrieval depth.
  2. Manually expand against a fixed repo list.
  3. Dynamically detect which repositories are represented in the initial candidate pool, then run a targeted second hybrid pass scoped to each.
* **Decision**: Option 3.
* **Implementation**: after the initial hybrid pass, active repositories are inferred from the returned payloads; a second hybrid query is run per detected repo and merged with the first pass, deduplicated by point ID.
* **Result**: see *Benchmark B*, "Hybrid + Repo Expansion + AST" column — the largest single improvement measured so far (Recall@10 0.33 → 0.50, MRR 0.28 → 0.34), at the cost of ~35% higher P95 latency from the extra scoped queries.


---
 
## 10. Neural Cross-Encoder Reranker: `BAAI/bge-reranker-base`
 
* **Context**: The heuristic `ASTAwareReranker` (Decision 9's rerank step) adjusts scores via path penalties, symbol-token matching, and call-graph bonuses — a legitimate but hand-tuned technique with no learned semantic understanding of the query/chunk relationship.
* **Decision**: Add a `diversified_cross` pipeline mode that scores the deduplicated, repo-expanded candidate pool with a real cross-encoder (`CrossEncoder("BAAI/bge-reranker-base")`, loaded once in `__init__`, called via `.predict()` over `[query, chunk_text]` pairs) instead of the heuristic reranker, evaluated as its own column rather than replacing the heuristic mode.
* **Result**: see *Benchmark C* below — Recall@10 improves to 0.483 (the highest recorded), but MRR (0.185) and Recall@1 (0.067) are the worst of any configuration measured, and P95 latency is 12,361 ms.
* **Latency assessment**: the cross-encoder is instantiated once, not per-query, so this is not a model-reload bug. The likely cause is unbatched, untruncated `CrossEncoder.predict()` calls over a wide candidate pool (up to `PRIMARY_POOL_LIMIT` + `EXPANSION_POOL_LIMIT` × detected repos, i.e. potentially 35–50 pairs per query) with each pair's chunk text going in at full length. This is a real, expected cost of cross-attention reranking at this pool size and precision, not a defect — but 12s is unusable for interactive use and needs addressing before this mode is production-viable: cap the pool fed to the cross-encoder, set `max_length` on `predict()`, and/or move inference to GPU.
* **Ranking-quality assessment**: still open — see Decision 12. Recall@10 rising while MRR and Recall@1 fall is not automatically evidence the cross-encoder is wrong; it may be scoring pool members it wasn't given before (post-expansion) that the heuristic reranker never saw. This needs to be separated from the open dense/hybrid discrepancies below before drawing a conclusion either way.
---

## Benchmark Results

Both runs use the 30-question benchmark from Decision 5.

### Benchmark A — Dense-Only (pre-hybrid baseline, generated 2026-06-25)

| Metric | Dense Baseline | Dense + AST Rerank | Dense + Repo Expansion + AST |
|:---|---:|---:|---:|
| MRR | 0.199 | 0.270 | 0.281 |
| Recall@1 | 0.150 | 0.233 | 0.250 |
| Recall@3 | 0.217 | 0.233 | 0.267 |
| Recall@5 | 0.233 | 0.233 | 0.267 |
| Recall@10 | 0.233 | 0.233 | 0.267 |
| P95 Latency (ms) | 116.0 | 82.9 | 206.1 |

This is the run referenced in Decision 6 — Recall@10 plateaus at 0.267 even with reranking and repo expansion layered on top of dense-only candidate generation, which is what pointed at candidate generation (not reranking) as the bottleneck.

### Benchmark B — Hybrid Retrieval (generated 2026-07-01)

| Metric | Hybrid (Dense + BM25 + RRF) Baseline | Hybrid + AST Rerank | Hybrid + Repo Expansion + AST |
|:---|---:|---:|---:|
| MRR | 0.221 | 0.281 | 0.335 |
| Recall@1 | 0.133 | 0.217 | 0.233 |
| Recall@3 | 0.217 | 0.217 | 0.250 |
| Recall@5 | 0.250 | 0.283 | 0.333 |
| Recall@10 | 0.333 | 0.450 | 0.500 |
| P95 Latency (ms) | 164.5 | 119.2 | 221.0 |

**Reading these honestly**: switching to hybrid candidate generation (Decision 8) lifted the Recall@10 ceiling from 0.267 to 0.333 before any reranking. Repo-aware expansion (Decision 9) then pushed it to 0.500 — the single biggest lever measured so far. But even in the best configuration, half of the golden-answer files still never appear in the top 10. That's still a candidate-generation limitation, not a reranking one, and it's the reason Decision 10 (named multi-vector storage) and a real cross-encoder reranker are the next priorities, ahead of any further heuristic tuning.

### Benchmark C — Cross-Encoder Trial (2026-07-01)
 
| Metric | Dense Baseline | Hybrid (Dense + BM25 + RRF) | Hybrid + AST | Hybrid + Repo Expansion + Cross-Encoder |
|:---|---:|---:|---:|---:|
| MRR | 0.278 | 0.239 | 0.281 | 0.185 |
| Recall@1 | 0.217 | 0.150 | 0.217 | 0.067 |
| Recall@3 | 0.317 | 0.217 | 0.217 | 0.183 |
| Recall@5 | 0.317 | 0.283 | 0.283 | 0.217 |
| Recall@10 | 0.367 | 0.333 | 0.450 | 0.483 |
| P95 Latency (ms) | 113.0 | 120.3 | 100.3 | 12,361.7 |
 
Column-by-column status: the final column's high latency is explained (Decision 10 — unbatched cross-attention over a wide pool, not a reload bug) but its ranking-quality result is not yet interpretable on its own, because the **Dense Baseline** and **Hybrid** columns in this same run don't reconcile with Benchmarks A/B.