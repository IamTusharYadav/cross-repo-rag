# Architecture Decisions Log (`decisions.md`)

This document records the critical architectural decisions, trade-offs, and technical solutions governing the data engineering, ingestion tracking, and vector indexing layers of the Cross-Repo RAG system.

---

## 1. Local Cache Invalidation Topology: Relational State Ledger via SQLite
* **Context**: Parsing and embedding large-scale engineering software repositories (e.g., `fastapi`, `qdrant-client`) introduces intense computational strain, high third-party embedding API costs, and vector collection drift if performed from scratch on every pipeline iteration.
* **Options Considered**:
    1.  **Stateless Full Re-Indexing**: Clear the database index and process every asset on every execution line. (Simple to build, but fails to scale linearly).
    2.  **In-Vector Payload Inspections**: Query the remote vector store at runtime to check if a specific path exists prior to document reading. (Creates a massive network call bottleneck and cannot reliably catch in-place file content modifications).
    3.  **Out-of-Band State Tracking Ledger (`SQLite`)**: Track repository file state locally via a persistent relational schema.
* **Decision**: Chosen **Option 3 (Out-of-Band SQLite State Ledger)**.
* **Justification**: Implemented via `src.db.state_manager.IngestionStateManager`. By tracking target records using an atomic `needs_processing()` method, the ingestion worker achieves highly efficient $O(1)$ local signature tracking. This bounds vector generation strictly to new or modified files, minimizing token spending and network roundtrips.

---

## 2. Ingestion Domain Separation: Orchestration vs. Vector Responsibilities
* **Context**: Bundling system logic—such as parsing, custom credential sanitization, AST splitting, tokenization budgets, and network transport drivers—into a single engine creates fragile, unmaintainable modules that are complex to unit-test.
* **Options Considered**:
    1.  **Monolithic Ingestion Script**: A singular pipeline handling file crawls, direct string cleansing, and raw database engine requests.
    2.  **Decoupled Orchestration Engine Layer**: Isolate raw vector data structures, array embedding loops, and bulk transfers into a backend-specific layer.
* **Decision**: Chosen **Option 2 (Decoupled Orchestration Engine Layer)**.
* **Justification**: The ingestion workspace strictly partitions responsibilities between `src/ingest.py` (which manages repository crawling, path tracking, regex-based secret redaction, and file-type chunking calls) and `src/db/vector_store.py` (which abstracts the target vector store SDK connection parameters and data models). This ensures that changing the target vector database vendor requires altering only the collection wrapper, keeping the ingestion workflow fully intact.

---

## 3. Vector Key Strategy: Deterministic Namespace-Driven UUID5
* **Context**: Target database vector stores require a unique identity key (uint64 or string UUID) for each item slot. Randomly generated primary keys (`UUID4`) prevent deterministic tracking, causing duplications if an identical file block is re-indexed.
* **Options Considered**:
    1.  **Auto-Incrementing Global Integer Sequences**: Maintaining a numeric counter inside the state machine database. (Breaks during parallel multi-process writes across separate code bases).
    2.  **Random String Values (`UUID4`)**: Generating a random token signature for every vector entry. (Turns system re-runs into permanent insertions, bloating database size).
    3.  **Deterministic Namespace Derivation (`UUID5`)**: Instantiating unique point IDs derived from a highly structured, stable string address unique to that chunk.
* **Decision**: Chosen **Option 3 (Deterministic UUID5 Point IDs)**.
* **Justification**: Implemented using Python's native `uuid.uuid5(uuid.NAMESPACE_DNS, semantic_chunk_id)` mapping inside the batch pipeline. This approach guarantees idempotent indexing behavior: repeating a run updates the exact same point slot without duplicate generation.

---

## 4. Chunk Tracking Integrity: Structural Signature vs. Stable Semantic Address
* **Context**: Source code is fluid; line numbers shift downward as changes are introduced. If vector storage identifiers are tied purely to line positions or simple sequence hashes, localized code edits can cause a cascade of broken vector identities throughout the remaining file blocks.
* **Options Considered**:
    1.  **Strict Hash Tracking (`chunk_id`)**: A unique SHA-256 string derived purely from a chunk's literal text. (Fragile to superficial syntax formatting updates or typo fixes).
    2.  **Stable Structural Addressing (`semantic_chunk_id`)**: Building a composite logical location coordinates pattern (`{repo}::{symbol_path}::chunk{idx}`).
* **Decision**: Co-implemented both identifiers as parallel, complementary metadata properties.
* **Justification**:
    * **`chunk_id`** (SHA-256): Represents a rigid snapshot signature of literal text, enabling the state ledger to quickly verify if content inside a file has physically shifted.
    * **`semantic_chunk_id`**: Establishes a logical location anchor (e.g., fallback pattern `{file_path}_{chunk_start_line}_{chunk_end_line}_{text}`). This provides downstream search workers and graph traversal agents with an unmoving, semantic coordinate link regardless of changing parent line indexes.

---

## 5. Storage Topology: Local Persistent Disk Directory vs. In-Memory Sub-Processes
* **Context**: Developing and testing data processing engines across large software products requires an index store that retains state between terminal sessions, without generating complex infrastructure or networking overhead.
* **Options Considered**:
    1.  **Ephemeral Engine (`:memory:`)**: Lightweight and configuration-free, but wipes all embedded collections as soon as the Python thread loop exits.
    2.  **Persistent Storage Directory**: Configure the vector backend client to write collection configurations, indexes, and dense matrices directly to a local path on disk.
* **Decision**: Chosen **Option 2 (Persistent Storage Directory)**.
* **Justification**: Configured via `QdrantClient(path="data/qdrant_storage")` (or an equivalent local file scheme via environment paths). This eliminates the friction of maintaining local Docker container layouts or cloud database subscriptions during initial development, while ensuring data persists across system restarts and standalone evaluation iterations.

---

## 6. Throughput Optimization: Explicit Payload Filter Indexing
* **Context**: Generic vector lookup algorithms compute cosine similarities across the entire collection space. As multi-repo indexing climbs to thousands of blocks, cross-talk between unrelated project layers degrades retrieval precision and introduces query latency.
* **Options Considered**:
    1.  **Post-Retrieval Filtering**: Extract a wide global vector array (`limit=100`), then programmatically prune out unwanted items using matching lists in memory. (Highly wasteful; burns database performance budget on irrelevant documents).
    2.  **Database Payload Keyword Indexing**: Direct the engine to build optimized search index maps over key filtering attributes during collection setup.
* **Decision**: Chosen **Option 2 (Database Payload Keyword Indexing)**.
* **Justification**: Managed during system setup by configuring payload indices over critical string keys like `repo_name` and `file_path`. This updates the database storage topology to process keyword constraints directly at the storage level, reducing query execution times down to single-digit milliseconds for tightly scoped repository queries.

---

## 7. Operational Resilience: Post-File Drift Pruning & Atomic State Processing
* **Context**: Code bases change; code blocks shrink or expand, and files are deleted. If a modified file produces fewer chunks than it did previously, the extra vector items from the older iteration remain orphaned in the database, introducing noise.
* **Design Implementation**:
    * **Atomicity Strategy**: In `src/ingest.py`, file tracking follows a strict sequence: chunk parsing $\rightarrow$ redaction scrubbing $\rightarrow$ text vectorization $\rightarrow$ vector upserting $\rightarrow$ stale point pruning. The file is marked as processed in SQLite *only* after all these stages succeed. If an error occurs, the state ledger remains un-updated, ensuring a clean retry on the next run.
    * **Drift Pruning**: `VectorStoreManager.prune_file_chunks()` evaluates the index using the array of active UUIDs generated during the current run. Chunks tied to the same file but missing from the active list are immediately pruned, eliminating old data remnants.
    * **Orphan Reconciliation**: A separate reconciliation function scans all records in the SQLite database against the file system. If a file is missing locally, its orphaned vectors are removed from the vector database, and its record is cleared from the state table.

---

## 8. Handling Ingestion Noise: The Evolution of the Test File Penalty
* **Context**: Baseline evaluation queries returned a high volume of test suite files (e.g., `fastapi/tests/...`, `qdrant-client/tests/...`). Because test suites contain hyper-dense concentrations of target implementation keywords and mock sub-invocations, they artificially outperform production files in standard dense vector space, creating a "Test File Swamp."
* **Why the Penalty Was Added**: Completely dropping test directories eliminates the RAG system's ability to answer developer queries regarding *how to test* the target architectures. Run-time penalization was selected to allow production source files to naturally claim the top ranks over test code while retaining full codebase context.
* **The Parametric Tuning Progression**:
    1.  **Initial Attempt (Penalty = `-0.15`)**: Successfully suppressed baseline test files on simple queries. However, for complex multi-repository questions, highly repetitive integration tests (such as `qdrant-client/tests/congruence_tests/test_updates.py`) still possessed baseline vector scores high enough to break through the filter, occupying the Rank 2 slot.
    2.  **Iterative Tuning (Penalty = `-0.30`)**: Doubled the penalty to actively drive test files out of the critical context generation window (Top 2 slots).
* **The Structural Result**: Increasing the penalty to `-0.30` broke the testing blockade. In the `DIVERSIFIED_AST` mode for the cross-repo query, the stubborn integration test (`test_updates.py`) was successfully demoted from Rank 2 down to Rank 3. This allowed a completely new, production-level utility module (`qdrant-client/tools/async_client_generator/fastembed_generator.py`) to rise to the surface at Rank 2.

---

## 9. Candidate Generation Optimization: Candidate Expansion Sweeps vs. Heuristic Overfitting
* **Context**: Multi-repository integration queries tracking asynchronous coordination boundary interfaces suffer from cross-repository semantic imbalance during dense vector retrieval. 
* **The Engineering Review Milestone**: Deep pool debugging (`eval_true_rank.py`) proved that target production files like `qdrant_client.py` were entirely absent from initial global retrieval pools up to a depth of 150 hits, while `applications.py` sat deep at Rank 29. This isolated the core system bottleneck to **Candidate Generation/Retrieval Quality** rather than Reranker weight configurations.
* **Rejected Alternative (Filename-Specific Boosting)**: Programmatic runtime score inflation for hardcoded signatures (e.g., boosting `applications.py` or `qdrant_client.py`) was explicitly rejected. While it artificializes benchmark metrics, it couples the retrieval model directly to naming conventions of specific projects, breaking generalizability to other software repositories.
* **The Implemented Strategy (Parametric Domain Expansion Sweep)**: To uncover hidden distributed dependencies safely without target leakage, the system executes a multi-stage candidate discovery pipeline:
  1. **Semantic Anchor Pass**: Run a baseline global dense query ($K=20$) to dynamically infer active repository namespaces.
  2. **Bounded Filtered Expansion**: Execute isolated, payload-filtered queries restricted to the discovered repository keys up to a swept limit ($L$).
  3. **Noise Truncation Window**: Merge and sort candidates via raw cosine similarity, discarding items outside the Top 50 to shield the downstream `ASTAwareReranker` from high-entropy vector noise.
* **Hyperparameter Matrix**:

| Configuration Strategy | Recall@1 | Recall@3 | Recall@5 | P95 Latency (ms) | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| Dense Baseline (Global) | 0.50 | 0.50 | 0.50 | ~291ms | Baseline Noise Ceiling |
| Repo Expansion (Limit=15) + AST | 0.50 | 0.50 | 0.50 | ~206ms | Insufficient Sweep Depth |
| Repo Expansion (Limit=30) + AST | *TBD* | *TBD* | *TBD* | *TBD* | Experimental Sweep Slot |
| Repo Expansion (Limit=45) + AST | *TBD* | *TBD* | *TBD* | *TBD* | Experimental Sweep Slot |
| Repo Expansion (Limit=60) + AST | *TBD* | *TBD* | *TBD* | *TBD* | Latency Bounds Test Block |