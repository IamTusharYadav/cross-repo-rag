import os
import json
import time
from dotenv import load_dotenv
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from src.search.heuristic_reranker import ASTAwareReranker

load_dotenv()


class RetrievalEvaluator:
    COLLECTION_NAME = "cross_repo_rag"

    # Primary Retrieval Phase Limits
    PRIMARY_PREFETCH_LIMIT = 60  # Depth of initial dense/sparse streams before fusion
    PRIMARY_POOL_LIMIT = 20  # Max candidates kept after initial RRF fusion

    # Domain Expansion Phase Limits
    EXPANSION_PREFETCH_LIMIT = (
        45  # Depth of per-repo dense/sparse streams before fusion
    )
    EXPANSION_POOL_LIMIT = 15  # Max candidates pulled per active repository domain

    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )

        # Dense Encoder
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

        # Sparse Encoder
        self.sparse_embed_model = SparseTextEmbedding(model_name="Qdrant/bm25")

        self.ast_reranker = ASTAwareReranker()

        print("Loading Neural Cross-Encoder Reranker (BAAI/bge-reranker-base)...")
        self.cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

        with open("eval/dataset_v1.json", "r") as f:
            self.dataset = json.load(f)

    def _calculate_metrics(
        self, expected_files: List[str], retrieved_paths: List[str]
    ) -> Dict[str, float]:
        metrics = {}

        def path_match(expected: str, retrieved: str) -> bool:
            exp_norm = expected.replace("\\", "/").lower()
            ret_norm = retrieved.replace("\\", "/").lower()
            return exp_norm in ret_norm or ret_norm.endswith(exp_norm)

        for k in [1, 3, 5, 10]:
            window = retrieved_paths[:k]
            matched = set(
                target
                for target in expected_files
                if any(path_match(target, r_path) for r_path in window)
            )
            metrics[f"recall_{k}"] = (
                len(matched) / len(expected_files) if expected_files else 0.0
            )

        mrr = 0.0
        for rank, r_path in enumerate(retrieved_paths, start=1):
            if any(path_match(target, r_path) for target in expected_files):
                mrr = 1.0 / rank
                break
        metrics["mrr"] = mrr
        return metrics

    def _encode_hybrid_query(
        self, text: str
    ) -> Tuple[List[float], models.SparseVector]:
        """
        Encodes query string into both normalized dense float arrays and
        structured sparse key-value paired tokens.
        """
        dense_res = self.embed_model.encode(
            text,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        sparse_raw = list(self.sparse_embed_model.embed([text]))[0]
        sparse_res = models.SparseVector(
            indices=sparse_raw.indices.tolist(),
            values=sparse_raw.values.tolist(),
        )
        return dense_res, sparse_res

    def _execute_query_layer(
        self,
        dense_vector: List[float],
        sparse_vector: Optional[models.SparseVector] = None,
        prefetch_limit: int = 60,
        pool_limit: int = 20,
        query_filter: Optional[models.Filter] = None,
        force_dense_only: bool = False,
    ) -> Any:
        """
        Unified Qdrant access layer wrapping prefetch matrix construction and execution logic.
        """
        # Legacy Baseline Isolation: Fetch purely from the dense vector subsystem
        if force_dense_only or sparse_vector is None:
            return self.qdrant_client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=dense_vector,
                using="dense",
                limit=pool_limit,
                query_filter=query_filter,
                with_payload=True,
            )

        # Native Hybrid Engine Execution with RRF Fusion
        return self.qdrant_client.query_points(
            collection_name=self.COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using="dense", limit=prefetch_limit
                ),
                models.Prefetch(
                    query=sparse_vector, using="sparse", limit=prefetch_limit
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=pool_limit,
            query_filter=query_filter,
            with_payload=True,
        )

    async def evaluate_pipeline(self, mode: str) -> pd.DataFrame:
        results = []
        print(f"\nRunning Pipeline Variant: [Mode = {mode.upper()}] ---")

        for item in self.dataset:
            start_time = time.perf_counter()

            # Hybrid Retrieval Pass (Global Semantic + Lexical Anchor)
            dense_vector, sparse_vector = await asyncio.to_thread(
                self._encode_hybrid_query, item["question"]
            )

            # Core Target Base Retrieval Execution
            is_dense_baseline = mode == "dense_baseline"
            response = await asyncio.to_thread(
                lambda: self._execute_query_layer(
                    dense_vector=dense_vector,
                    sparse_vector=sparse_vector,
                    prefetch_limit=self.PRIMARY_PREFETCH_LIMIT,
                    pool_limit=self.PRIMARY_POOL_LIMIT,
                    force_dense_only=is_dense_baseline,
                )
            )
            initial_hits = response.points if hasattr(response, "points") else response
            final_pool = list(initial_hits)

            # Structural Repository Domain Expansion
            if mode in ["diversified_ast", "diversified_cross"]:
                # Detect active repositories represented in the semantic anchor pool
                detected_repos = set()
                for hit in initial_hits:
                    payload = (
                        hit.payload
                        if hasattr(hit, "payload")
                        else hit.get("payload", {})
                    )
                    repo = payload.get("repo_name")
                    if repo:
                        detected_repos.add(repo)

                # Expand candidate pool around discovered repository domains dynamically
                expanded_hits = []
                for repo in detected_repos:
                    repo_response = await asyncio.to_thread(
                        lambda: self._execute_query_layer(
                            dense_vector=dense_vector,
                            sparse_vector=sparse_vector,
                            prefetch_limit=self.EXPANSION_PREFETCH_LIMIT,
                            pool_limit=self.EXPANSION_POOL_LIMIT,
                            query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="repo_name",
                                        match=models.MatchValue(value=repo),
                                    )
                                ]
                            ),
                        )
                    )
                    repo_points = (
                        repo_response.points
                        if hasattr(repo_response, "points")
                        else repo_response
                    )
                    expanded_hits.extend(repo_points)

                # Deduplicate merged candidates by point ID to avoid duplicate scoring
                seen_ids = set()
                deduplicated_pool = []
                for h in final_pool + expanded_hits:
                    if h.id not in seen_ids:
                        seen_ids.add(h.id)
                        deduplicated_pool.append(h)
                final_pool = deduplicated_pool

            # Ranking/Sorting Selection
            if mode == "hybrid_ast":
                processed_hits = self.ast_reranker.rerank(item["question"], final_pool)
            elif mode == "diversified_cross":
                # Deep Cross-Attention Scoring Phase
                query_text = item["question"]
                pairs = []
                for h in final_pool:
                    payload = (
                        h.payload if hasattr(h, "payload") else h.get("payload", {})
                    )
                    # Pull raw text preserved by the VectorStoreManager payload update
                    chunk_text = payload.get("text_content", "")
                    pairs.append([query_text, chunk_text])

                # Execute neural scoring off-thread to maintain event-loop responsiveness
                scores = await asyncio.to_thread(
                    lambda: self.cross_encoder.predict(pairs).tolist() if pairs else []
                )

                processed_hits = []
                for idx, h in enumerate(final_pool):
                    payload = (
                        h.payload if hasattr(h, "payload") else h.get("payload", {})
                    )
                    processed_hits.append(
                        {
                            "id": h.id,
                            "score": scores[idx],
                            "adjusted_score": scores[idx],
                            "payload": payload,
                        }
                    )
                processed_hits = sorted(
                    processed_hits, key=lambda x: x["score"], reverse=True
                )
            else:
                # Hybrid Baseline Ranking Sort
                processed_hits = [
                    {
                        "id": h.id,
                        "score": h.score,
                        "adjusted_score": h.score,
                        "payload": h.payload
                        if hasattr(h, "payload")
                        else h.get("payload", {}),
                    }
                    for h in sorted(final_pool, key=lambda x: x.score, reverse=True)
                ]

            latency_ms = (time.perf_counter() - start_time) * 1000

            retrieved_paths = [
                hit["payload"].get("file_path", "") for hit in processed_hits
            ]

            # Diagnostic Prints
            print(f"\nQUERY: {item['question']}")
            print(f"EXPECTED: {item['expected_files']}")
            print("RETRIEVED TOP 3:")
            for rank, r_path in enumerate(retrieved_paths[:3], start=1):
                print(f"  {rank}. {r_path}")

            metrics = self._calculate_metrics(item["expected_files"], retrieved_paths)
            results.append(
                {"query_id": item["id"], "latency_ms": latency_ms, **metrics}
            )

        return pd.DataFrame(results)

    async def run(self):
        print("Initiating Comparative Ablation Benchmark Run...")

        df_dense = await self.evaluate_pipeline(mode="dense_baseline")
        df_hybrid = await self.evaluate_pipeline(mode="hybrid_baseline")
        df_ast = await self.evaluate_pipeline(mode="hybrid_ast")
        df_cross = await self.evaluate_pipeline(mode="diversified_cross")

        summary = {
            "Metric": [
                "MRR",
                "Recall@1",
                "Recall@3",
                "Recall@5",
                "Recall@10",
                "P95 Latency (ms)",
            ],
            "Dense Baseline": [
                df_dense["mrr"].mean(),
                df_dense["recall_1"].mean(),
                df_dense["recall_3"].mean(),
                df_dense["recall_5"].mean(),
                df_dense["recall_10"].mean(),
                np.percentile(df_dense["latency_ms"], 95),
            ],
            "Hybrid (Dense + BM25 + RRF)": [
                df_hybrid["mrr"].mean(),
                df_hybrid["recall_1"].mean(),
                df_hybrid["recall_3"].mean(),
                df_hybrid["recall_5"].mean(),
                df_hybrid["recall_10"].mean(),
                np.percentile(df_hybrid["latency_ms"], 95),
            ],
            "Hybrid + AST": [
                df_ast["mrr"].mean(),
                df_ast["recall_1"].mean(),
                df_ast["recall_3"].mean(),
                df_ast["recall_5"].mean(),
                df_ast["recall_10"].mean(),
                np.percentile(df_ast["latency_ms"], 95),
            ],
            "Hybrid + Repo Expansion + Cross-Encoder": [
                df_cross["mrr"].mean(),
                df_cross["recall_1"].mean(),
                df_cross["recall_3"].mean(),
                df_cross["recall_5"].mean(),
                df_cross["recall_10"].mean(),
                np.percentile(df_cross["latency_ms"], 95),
            ],
        }

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv("eval/benchmark_results.csv", index=False)

        report_md = f"""# Retrieval Layer Ablation Benchmark Report
Generated on: 2026-07-01

## Aggregated Pipeline Performance Comparison
{df_summary.to_markdown(index=False)}
"""
        with open("eval/report.md", "w") as f:
            f.write(report_md)

        print("Benchmarks successfully executed. Results compiled in 'eval/report.md'.")


if __name__ == "__main__":
    asyncio.run(RetrievalEvaluator().run())
