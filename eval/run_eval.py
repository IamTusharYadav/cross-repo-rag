import os
import json
import time
from dotenv import load_dotenv
import asyncio
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from src.search.heuristic_reranker import ASTAwareReranker

load_dotenv()


class RetrievalEvaluator:
    def __init__(self):
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )
        self.embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        self.ast_reranker = ASTAwareReranker()

        with open("eval/golden_questions.json", "r") as f:
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

    async def evaluate_pipeline(self, mode: str) -> pd.DataFrame:
        results = []
        print(f"\nRunning Pipeline Variant: [Mode = {mode.upper()}] ---")

        for item in self.dataset:
            start_time = time.perf_counter()

            # Stage 1: Dense Retrieval Pass (Global Semantic Anchor)
            query_vector = await asyncio.to_thread(
                lambda: self.embed_model.encode(item["question"]).tolist()
            )

            # Initial candidate pool
            response = await asyncio.to_thread(
                lambda: self.qdrant_client.query_points(
                    collection_name="cross_repo_rag",
                    query=query_vector,
                    limit=20,
                    with_payload=True,
                )
            )
            initial_hits = response.points if hasattr(response, "points") else response

            final_pool = list(initial_hits)

            # Stage 2: Repository-Aware Candidate Expansion (Production Variant)
            if mode == "diversified_ast":
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
                        lambda: self.qdrant_client.query_points(
                            collection_name="cross_repo_rag",
                            query=query_vector,
                            query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="repo_name",
                                        match=models.MatchValue(value=repo),
                                    )
                                ]
                            ),
                            limit=15,  # Dig deep into each identified domain
                            with_payload=True,
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

            # Stage 3: Ranking/Sorting Selection
            # DEBUGGING BLOCK
            if mode == "diversified_ast" and item["id"] == "q1":
                print(f"\nDEBUGGING DIVERSIFIED POOL FOR QUESTION 1")
                print(f"Total points collected in final_pool: {len(final_pool)}")
                print("All file paths present in the pool:")
                for idx, h in enumerate(final_pool):
                    p = h.payload if hasattr(h, "payload") else h.get("payload", {})
                    print(f"  [{idx}] {p.get('file_path')}")
            if mode in ["ast_only", "diversified_ast"]:
                processed_hits = self.ast_reranker.rerank(item["question"], final_pool)
            else:
                # Dense Baseline Sorting
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
        print("⚡ Initiating Comparative Ablation Benchmark Run...")

        df_base = await self.evaluate_pipeline(mode="baseline")
        df_ast = await self.evaluate_pipeline(mode="ast_only")
        df_div = await self.evaluate_pipeline(mode="diversified_ast")

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
                df_base["mrr"].mean(),
                df_base["recall_1"].mean(),
                df_base["recall_3"].mean(),
                df_base["recall_5"].mean(),
                df_base["recall_10"].mean(),
                np.percentile(df_base["latency_ms"], 95),
            ],
            "Dense + AST Rerank": [
                df_ast["mrr"].mean(),
                df_ast["recall_1"].mean(),
                df_ast["recall_3"].mean(),
                df_ast["recall_5"].mean(),
                df_ast["recall_10"].mean(),
                np.percentile(df_ast["latency_ms"], 95),
            ],
            "Dense + Repo Expansion + AST": [
                df_div["mrr"].mean(),
                df_div["recall_1"].mean(),
                df_div["recall_3"].mean(),
                df_div["recall_5"].mean(),
                df_div["recall_10"].mean(),
                np.percentile(df_div["latency_ms"], 95),
            ],
        }

        df_summary = pd.DataFrame(summary)
        df_summary.to_csv("eval/benchmark_results.csv", index=False)

        report_md = f"""# Retrieval Layer Ablation Benchmark Report
Generated on: 2026-06-25

## Aggregated Pipeline Performance Comparison
{df_summary.to_markdown(index=False)}
"""
        with open("eval/report.md", "w") as f:
            f.write(report_md)

        print("Benchmarks successfully executed. Results compiled in 'eval/report.md'.")


if __name__ == "__main__":
    asyncio.run(RetrievalEvaluator().run())
