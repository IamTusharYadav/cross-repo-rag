import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tiktoken
from src.db.state_manager import IngestionStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
)
logger = logging.getLogger(__name__)


class MarkdownHierarchyChunker:
    """
    A structural Markdown chunker featuring:
    - Line-coordinate tracking with zero coordinate drift.
    - Tokenization memoization caching for high-throughput multi-repo parsing.
    - Stateful code-fence healing across text chunk boundaries.
    - Explicit self-contained payload header normalization.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_percent: float = 0.15,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = int(max_tokens * overlap_percent)
        self.embedding_model = embedding_model
        # Ingestion throughput optimization cache
        self._token_cache: Dict[str, int] = {}

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "cl100k_base encoding missing, falling back to gpt-4 tokenizer engine."
            )
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def _get_token_len(self, text: str) -> int:
        """Retrieves token count from a memoized dictionary cache to maximize CPU performance."""
        if text not in self._token_cache:
            self._token_cache[text] = len(self.tokenizer.encode(text))
        return self._token_cache[text]

    def _line_aware_sliding_window(
        self,
        parsed_lines: List[Dict[str, Any]],
        heading_context: str,
        section_name: str,
        section_start_line: int,
    ) -> List[Dict[str, Any]]:
        """
        Slices structured section lines while protecting code blocks from broken syntax.
        Auto-injects closing and opening markdown code fences at window junctions.
        """
        sub_chunks = []

        # Standardized, explicit payload headers for isolated chunk retrieval resilience
        breadcrumb_prefix = f"Context: {heading_context}\n" if heading_context else ""
        section_prefix = (
            f"Section: {section_name}\n----------------------------------------\n"
        )
        meta_header = breadcrumb_prefix + section_prefix

        prefix_len = self._get_token_len(meta_header)
        usable_max_tokens = self.max_tokens - prefix_len

        if usable_max_tokens <= 60:
            usable_max_tokens = max(100, self.max_tokens - 60)

        current_chunk_lines: List[str] = []
        current_chunk_tokens = 0
        line_offsets: List[int] = []

        idx = 0
        while idx < len(parsed_lines):
            line_item = parsed_lines[idx]
            raw_line_text = line_item["text"]
            line_token_len = self._get_token_len(raw_line_text)

            # Safeguard for anomalous, ultra-long string blocks
            if line_token_len > usable_max_tokens:
                if current_chunk_lines:
                    sub_chunks.append(
                        {
                            "text": meta_header + "".join(current_chunk_lines),
                            "start_line": section_start_line + line_offsets[0],
                            "end_line": section_start_line + line_offsets[-1],
                            "token_count": current_chunk_tokens + prefix_len,
                        }
                    )
                    current_chunk_lines, line_offsets, current_chunk_tokens = [], [], 0

                sub_chunks.append(
                    {
                        "text": meta_header + raw_line_text,
                        "start_line": section_start_line + idx,
                        "end_line": section_start_line + idx,
                        "token_count": line_token_len + prefix_len,
                    }
                )
                idx += 1
                continue

            # Check if adding the next row bursts the current token window budget
            if current_chunk_tokens + line_token_len > usable_max_tokens:
                # Stateful Code Fence Healing Logic
                # If we must split while inside an active block, heal the layout structure
                heal_suffix = ""
                active_fence_at_break = line_item["active_fence"]

                if active_fence_at_break is not None:
                    heal_suffix = f"\n{active_fence_at_break}\n"

                payload_text = meta_header + "".join(current_chunk_lines) + heal_suffix

                sub_chunks.append(
                    {
                        "text": payload_text,
                        "start_line": section_start_line + line_offsets[0],
                        "end_line": section_start_line + line_offsets[-1],
                        "token_count": self._get_token_len(payload_text),
                    }
                )

                # Process sliding window backtracking logic
                overlap_lines_accumulated: List[str] = []
                overlap_tokens_accumulated = 0
                backtrack_count = 0

                for back_idx in range(len(current_chunk_lines) - 1, -1, -1):
                    back_line = current_chunk_lines[back_idx]
                    back_token_len = self._get_token_len(back_line)

                    if (
                        overlap_tokens_accumulated + back_token_len
                        > self.overlap_tokens
                    ):
                        break
                    overlap_lines_accumulated.insert(0, back_line)
                    overlap_tokens_accumulated += back_token_len
                    backtrack_count += 1

                if backtrack_count > 0:
                    current_chunk_lines = overlap_lines_accumulated
                    line_offsets = line_offsets[-backtrack_count:]
                    current_chunk_tokens = overlap_tokens_accumulated
                else:
                    current_chunk_lines, line_offsets, current_chunk_tokens = [], [], 0

                # Prepend the open code fence identifier to the fresh chunk window
                if active_fence_at_break is not None and line_item["fence_header_line"]:
                    header_injector = f"{line_item['fence_header_line']}\n"
                    current_chunk_lines.insert(0, header_injector)
                    current_chunk_tokens += self._get_token_len(header_injector)

            current_chunk_lines.append(raw_line_text)
            line_offsets.append(idx)
            current_chunk_tokens += line_token_len
            idx += 1

        if current_chunk_lines:
            payload_text = meta_header + "".join(current_chunk_lines)
            sub_chunks.append(
                {
                    "text": payload_text,
                    "start_line": section_start_line + line_offsets[0],
                    "end_line": section_start_line + line_offsets[-1],
                    "token_count": self._get_token_len(payload_text),
                }
            )

        return sub_chunks

    def _extract_repo_name(self, file_path: str) -> str:
        """Identifies parent repository workspace folder names accurately."""
        normalized_path = os.path.normpath(file_path)
        parts_path = normalized_path.split(os.sep)
        if "data" in parts_path:
            data_index = parts_path.index("data")
            if data_index + 1 < len(parts_path):
                extracted = parts_path[data_index + 1]
                if extracted.endswith((".md", ".py", ".txt", ".json")):
                    return "root-docs"
                return extracted
        return "unknown"

    def chunk_file(
        self,
        file_path: str,
        state_manager: IngestionStateManager = None,
        repo_root: str = "./data",
    ) -> List[Dict[str, Any]]:
        """Parses Markdown files hierarchically while preserving raw physical line numbering perfectly."""
        chunks = []
        if not os.path.isfile(file_path):
            logger.error(f"Markdown verification failure. File absent: {file_path}")
            return chunks

        repo_name = self._extract_repo_name(file_path)
        rel_path = os.path.relpath(file_path)
        path_obj = Path(rel_path)

        if state_manager is not None:
            if not state_manager.needs_processing(file_path, repo_root, repo_name):
                logger.info(
                    f"Markdown cache hit. Skipping parsing operations for: {rel_path}"
                )
                return chunks

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_lines = f.readlines()
        except Exception as e:
            logger.error(
                f"Failed to access Markdown text structures for {file_path}: {e}"
            )
            return chunks

        # LINE-PRESERVING PREPROCESSING PASS

        cleaned_lines: List[str] = []
        in_logo_html_block = False

        p_start_pattern = re.compile(r"<p\s+align=")
        p_end_pattern = re.compile(r"</p>")
        standalone_image_pattern = re.compile(r"^!\[.*?\]\(.*?\)$")

        for current_idx, line in enumerate(raw_lines):
            stripped = line.strip()

            if p_start_pattern.search(line):
                lookahead_limit = min(len(raw_lines), current_idx + 6)
                lookahead_window = "".join(
                    raw_lines[current_idx:lookahead_limit]
                ).lower()

                if (
                    "img" in lookahead_window
                    or "logo" in lookahead_window
                    or "badge" in lookahead_window
                ):
                    in_logo_html_block = True
                    cleaned_lines.append("\n")
                    continue

            if in_logo_html_block:
                if p_end_pattern.search(line):
                    in_logo_html_block = False
                cleaned_lines.append("\n")
                continue

            if stripped.startswith("[![") or standalone_image_pattern.match(stripped):
                cleaned_lines.append("\n")
                continue

            cleaned_lines.append(line)

        # (heading_level, heading_text)
        current_headings: List[Tuple[int, str]] = []
        current_section_lines: List[Dict[str, Any]] = []
        section_start_line = 1

        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
        active_code_fence: Optional[str] = None
        current_fence_header_line: Optional[str] = None

        def process_accumulated_section(
            section_data_lines: List[Dict[str, Any]], start_line: int
        ):
            if not section_data_lines:
                return

            # Heading deduplication step
            if section_data_lines and header_pattern.match(
                section_data_lines[0]["text"]
            ):
                section_data_lines = section_data_lines[1:]
                start_line += 1

            if not section_data_lines:
                return

            heading_hierarchy_str = " > ".join([text for _, text in current_headings])
            current_node_name = (
                current_headings[-1][1] if current_headings else "Root Module"
            )
            parent_node_name = (
                current_headings[-2][1] if len(current_headings) > 1 else None
            )

            windows = self._line_aware_sliding_window(
                section_data_lines, heading_hierarchy_str, current_node_name, start_line
            )
            total_sec_chunks = len(windows)

            for idx, win in enumerate(windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": {
                            "repo_name": repo_name,
                            "file_path": rel_path,
                            "filename": path_obj.name,
                            "module_name": path_obj.stem,
                            "language": "markdown",
                            "extension": ".md",
                            "node_type": "markdown_section",
                            "chunk_type": "documentation",
                            "name": current_node_name,
                            "parent": parent_node_name,
                            "parent_type": "markdown_section"
                            if parent_node_name
                            else None,
                            "start_line": start_line,
                            "end_line": start_line
                            + max(0, len(section_data_lines) - 1),
                            "chunk_start_line": win["start_line"],
                            "chunk_end_line": win["end_line"],
                            "chunk_index": idx,
                            "chunk_count": total_sec_chunks,
                            "token_count": win["token_count"],
                            "embedding_model": self.embedding_model,
                            "chunking_strategy": "StructuralMarkdownHierarchy",
                        },
                    }
                )

        # CORE PARSING ENGINE LOOP
        for current_idx, line in enumerate(cleaned_lines, start=1):
            stripped = line.strip()

            if stripped.startswith(("```", "~~~")):
                detected_fence = "```" if stripped.startswith("```") else "~~~"

                if active_code_fence == detected_fence:
                    # Closing active block
                    current_section_lines.append(
                        {
                            "text": line,
                            "active_fence": active_code_fence,
                            "fence_header_line": current_fence_header_line,
                        }
                    )
                    active_code_fence = None
                    current_fence_header_line = None
                elif active_code_fence is None:
                    # Opening new block
                    active_code_fence = detected_fence
                    current_fence_header_line = stripped
                    current_section_lines.append(
                        {
                            "text": line,
                            "active_fence": active_code_fence,
                            "fence_header_line": current_fence_header_line,
                        }
                    )
                else:
                    # Nested code-fence mismatch edge-case handled as normal internal text
                    current_section_lines.append(
                        {
                            "text": line,
                            "active_fence": active_code_fence,
                            "fence_header_line": current_fence_header_line,
                        }
                    )
                continue

            if active_code_fence is None:
                match = header_pattern.match(line)
                if match:
                    process_accumulated_section(
                        current_section_lines, section_start_line
                    )
                    current_section_lines = []
                    section_start_line = current_idx

                    level = len(match.group(1))
                    title = match.group(2).strip()

                    while current_headings and current_headings[-1][0] >= level:
                        current_headings.pop()

                    current_headings.append((level, title))
                    current_section_lines.append(
                        {"text": line, "active_fence": None, "fence_header_line": None}
                    )
                else:
                    current_section_lines.append(
                        {"text": line, "active_fence": None, "fence_header_line": None}
                    )
            else:
                current_section_lines.append(
                    {
                        "text": line,
                        "active_fence": active_code_fence,
                        "fence_header_line": current_fence_header_line,
                    }
                )

        # Flush final trailing text block remaining in buffer
        process_accumulated_section(current_section_lines, section_start_line)

        if state_manager is not None and chunks:
            state_manager.mark_as_processed(file_path, repo_root, repo_name)
            logger.info(
                f"Recorded state ledger signature safely for markdown: {rel_path}"
            )

        return chunks


if __name__ == "__main__":
    logger.info("Executing Markdown structural parser diagnostic run...")

    # Simple setup test files mock
    test_md_path = "data/qdrant-client/README.md"

    manager = IngestionStateManager("sqlite:///data/test_ledger.db")
    chunker = MarkdownHierarchyChunker(max_tokens=512, overlap_percent=0.10)

    extracted_chunks = chunker.chunk_file(
        test_md_path, state_manager=manager, repo_root="."
    )
    print(
        f"\nDiscovered {len(extracted_chunks)} structural blocks inside the Markdown parser."
    )

    for i, chk in enumerate(extracted_chunks):
        print(f"\n{'=' * 70}\nChunk Block {i}\n{'=' * 70}")
        print("Flat Metadata:")
        for k, v in chk["metadata"].items():
            print(f"  {k}: {v}")
        print("\nPayload Body Context String Preview:")
        print("-" * 40)
        print(chk["text"])
        print("-" * 40)
