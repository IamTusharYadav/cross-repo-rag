import re
from typing import List, Dict, Any


class ASTAwareReranker:
    """
    Executes zero-latency structural ranking adjustments based on AST symbols,
    avoiding the latency and computational cost of neural cross-encoders.
    """

    def __init__(self):
        # Identify if the user's intent is explicitly looking for code structures
        self.structural_keywords = re.compile(
            r"\b(class|function|method|def|async)\b", re.IGNORECASE
        )

    def rerank(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        has_structural_intent = bool(self.structural_keywords.search(query_lower))
        query_words = set(re.findall(r"[a-z0-9_]{3,}", query_lower))

        reranked_hits = []
        for hit in hits:
            payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
            score = hit.score if hasattr(hit, "score") else hit.get("score", 0.0)

            file_path = payload.get("file_path", "").lower()
            node_type = payload.get("node_type", "").lower()
            symbol_path = payload.get("symbol_path", "").lower()
            calls = [c.lower() for c in payload.get("calls", [])]

            # Test File De-boosting / Production Prioritization
            # Prevents test suites from swamping real implementation code
            if "test" in file_path or "tests" in file_path:
                score -= 0.30

            # Intent-Based Boosting
            if has_structural_intent:
                if node_type in ["function", "method"]:
                    score += 0.08
                elif node_type == "class":
                    score += 0.06

            # Accumulative Symbol Path Token Matching
            if symbol_path:
                symbol_tokens = [
                    t for t in re.split(r"[\._:]", symbol_path) if len(t) > 2
                ]
                match_count = 0
                for token in symbol_tokens:
                    if token in query_lower or token in query_words:
                        match_count += 1

                if match_count > 0:
                    score += min(0.05 * match_count, 0.15)

            # Call-Graph Sub-Invocation Validation
            matching_calls = 0
            for call in calls:
                clean_call = call.split(".")[-1]
                if len(clean_call) > 2:
                    if clean_call in query_lower or clean_call in query_words:
                        matching_calls += 1

            if matching_calls > 0:
                score += min(0.03 * matching_calls, 0.12)

            processed_hit = {
                "id": hit.id if hasattr(hit, "id") else hit.get("id"),
                "score": hit.score if hasattr(hit, "score") else hit.get("score", 0.0),
                "adjusted_score": score,
                "payload": payload,
            }
            reranked_hits.append(processed_hit)

        # Re-sort descending by our metadata-optimized metric
        reranked_hits.sort(key=lambda x: x["adjusted_score"], reverse=True)
        return reranked_hits
