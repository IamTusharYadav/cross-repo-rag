import ast
import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import tiktoken
from src.db.state_manager import IngestionStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
)
logger = logging.getLogger(__name__)


class PythonASTChunker:
    """
    A production-grade Abstract Syntax Tree (AST) chunker that parses Python source files,
    isolates meaningful logical code blocks (modules, functions, classes, methods), tracks hierarchical
    parent-child relationships and maps precise line boundries for accurate chunk-level source highlighting.
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

        # Intialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "cl100k_base encoding not found, falling back to gpt-4 tokenizer."
            )
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    def _get_node_source(self, source_code: str, node: ast.AST) -> str:
        """Extracts the raw source code string of a given AST node."""
        try:
            segment = ast.get_source_segment(source_code, node)
            if segment:
                return segment
        except Exception:
            pass

        # Fallback line-slicing
        lines = source_code.splitlines()
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", len(lines))
        return "\n".join(lines[start_line:end_line])

    def _split_by_token_window(
        self, text: str, node_start_line: int
    ) -> List[Dict[str, Any]]:
        """
        Slices text strings into strict token-bounded arrays while computing localized
        chunk_start_line and chunk_end_line boundaries dynamically based on character evaluation.
        """
        tokens = self.tokenizer.encode(text)

        # content fits comfortably within a single chunk window
        if len(tokens) <= self.max_tokens:
            lines_in_text = text.splitlines()
            return [
                {
                    "text": text,
                    "chunk_start_line": node_start_line,
                    "chunk_end_line": node_start_line + max(0, len(lines_in_text) - 1),
                    "token_count": len(tokens),
                }
            ]

        sub_chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.max_tokens
            chunk_tokens = tokens[start:end]
            decoded_text = self.tokenizer.decode(chunk_tokens)

            # Count precise physical newlines to find line-mapping coordinates
            prefix_text = self.tokenizer.decode(tokens[:start])
            start_line_offset = prefix_text.count("\n")
            chunk_lines_count = decoded_text.count("\n")

            chunk_start = node_start_line + start_line_offset
            chunk_end = chunk_start + chunk_lines_count

            sub_chunks.append(
                {
                    "text": decoded_text,
                    "chunk_start_line": chunk_start,
                    "chunk_end_line": chunk_end,
                    "token_count": len(chunk_tokens),
                }
            )

            start += self.max_tokens - self.overlap_tokens
            if self.max_tokens <= self.overlap_tokens:
                break  # Security block to completely prevent infinite runtime loop parameters

        return sub_chunks

    def _extract_repo_name(self, file_path: str) -> str:
        """Identifies the repository name from the file path."""
        normalized_path = os.path.normpath(file_path)
        parts_path = normalized_path.split(os.sep)
        if "data" in parts_path:
            data_index = parts_path.index("data")
            if data_index + 1 < len(parts_path):
                return parts_path[data_index + 1]
        return "unknown"

    def chunk_file(
        self,
        file_path: str,
        state_manager: IngestionStateManager = None,
        repo_root: str = "./data",
    ) -> List[Dict[str, Any]]:
        """
        Parses a python file into structural, token-bounded chunks enriched with
        a flat production metadata schema. Uses IngestionStateManager to skip modified files.
        """
        chunks = []
        if not os.path.isfile(file_path):
            logger.error(f"File target structural absence flagged: {file_path}")
            return chunks

        repo_name = self._extract_repo_name(file_path)
        rel_path = os.path.relpath(file_path)
        path_obj = Path(rel_path)

        # STATE LEDGER DELTA CHECK
        if state_manager is not None:
            # Check if file state matches existing hashes to save indexing latency
            if not state_manager.needs_processing(file_path, repo_root, repo_name):
                logger.info(
                    f"File cache hit. Skipping structural parsing for: {rel_path}"
                )
                return chunks

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except Exception as e:
            logger.error(
                f"Failed to extract text data layers from file {file_path}: {e}"
            )
            return chunks

        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            logger.warning(
                f"Syntax anomaly skipped AST generation for {file_path}: {e}"
            )
            return chunks

        # MODULE DOCSTRING EXTRACTION PASS
        try:
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                doc_text = f'"""\n{module_docstring}\n"""'
                windows = self._split_by_token_window(doc_text, node_start_line=1)
                total_doc_chunks = len(windows)

                for idx, win in enumerate(windows):
                    chunks.append(
                        {
                            "text": win["text"],
                            "metadata": {
                                # Identity Segment
                                "repo_name": repo_name,
                                "file_path": rel_path,
                                "filename": path_obj.name,
                                "module_name": path_obj.stem,
                                "language": "python",
                                "extension": ".py",
                                # Structural/Location Segment
                                "node_type": "module_docstring",
                                "chunk_type": "docstring",
                                "name": path_obj.stem,
                                "parent": None,
                                "parent_type": None,
                                "start_line": 1,
                                "end_line": len(doc_text.splitlines()),
                                "chunk_start_line": win["chunk_start_line"],
                                "chunk_end_line": win["chunk_end_line"],
                                "chunk_index": idx,
                                "chunk_count": total_doc_chunks,
                                # Indexing Performance Segment
                                "token_count": win["token_count"],
                                "embedding_model": self.embedding_model,
                                "chunking_strategy": "HierarchicalAST",
                            },
                        }
                    )
        except Exception as e:
            logger.error(
                f"Error compiling module level documentation segments in {file_path}: {e}"
            )

        # Track global syntax assignments (imports, configuration maps, constants)
        global_elements = []
        global_start_line = None

        # ITERATIVE NODE PARSING ENGINES
        for node in tree.body:
            try:
                # Top-Level Code Functions
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_code = self._get_node_source(source_code, node)
                    windows = self._split_by_token_window(
                        func_code, node_start_line=node.lineno
                    )
                    total_chunks = len(windows)

                    for idx, win in enumerate(windows):
                        chunks.append(
                            {
                                "text": win["text"],
                                "metadata": {
                                    "repo_name": repo_name,
                                    "file_path": rel_path,
                                    "filename": path_obj.name,
                                    "module_name": path_obj.stem,
                                    "language": "python",
                                    "extension": ".py",
                                    "node_type": "function",
                                    "chunk_type": "code_ast",
                                    "name": node.name,
                                    "parent": None,
                                    "parent_type": None,
                                    "start_line": node.lineno,
                                    "end_line": getattr(
                                        node, "end_lineno", node.lineno
                                    ),
                                    "chunk_start_line": win["chunk_start_line"],
                                    "chunk_end_line": win["chunk_end_line"],
                                    "chunk_index": idx,
                                    "chunk_count": total_chunks,
                                    "token_count": win["token_count"],
                                    "embedding_model": self.embedding_model,
                                    "chunking_strategy": "HierarchicalAST",
                                },
                            }
                        )

                # Class Implementations and Nested System Methods
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_code = self._get_node_source(source_code, node)
                    windows = self._split_by_token_window(
                        class_code, node_start_line=node.lineno
                    )
                    total_class_chunks = len(windows)

                    # Extract Holistic Class Framework Blueprint
                    for idx, win in enumerate(windows):
                        chunks.append(
                            {
                                "text": win["text"],
                                "metadata": {
                                    "repo_name": repo_name,
                                    "file_path": rel_path,
                                    "filename": path_obj.name,
                                    "module_name": path_obj.stem,
                                    "language": "python",
                                    "extension": ".py",
                                    "node_type": "class",
                                    "chunk_type": "code_ast",
                                    "name": class_name,
                                    "parent": None,
                                    "parent_type": None,
                                    "start_line": node.lineno,
                                    "end_line": getattr(
                                        node, "end_lineno", node.lineno
                                    ),
                                    "chunk_start_line": win["chunk_start_line"],
                                    "chunk_end_line": win["chunk_end_line"],
                                    "chunk_index": idx,
                                    "chunk_count": total_class_chunks,
                                    "token_count": win["token_count"],
                                    "embedding_model": self.embedding_model,
                                    "chunking_strategy": "HierarchicalAST",
                                },
                            }
                        )

                    # Traverse class contents to extract independent method segments
                    for sub_node in node.body:
                        if isinstance(
                            sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            method_code = self._get_node_source(source_code, sub_node)
                            contextual_code = f"class {class_name}:\n    {method_code}"

                            method_windows = self._split_by_token_window(
                                contextual_code, node_start_line=sub_node.lineno
                            )
                            total_method_chunks = len(method_windows)

                            for idx, win in enumerate(method_windows):
                                chunks.append(
                                    {
                                        "text": win["text"],
                                        "metadata": {
                                            "repo_name": repo_name,
                                            "file_path": rel_path,
                                            "filename": path_obj.name,
                                            "module_name": path_obj.stem,
                                            "language": "python",
                                            "extension": ".py",
                                            "node_type": "method",
                                            "chunk_type": "code_ast",
                                            "name": f"{class_name}.{sub_node.name}",
                                            "parent": class_name,
                                            "parent_type": "class",
                                            "start_line": sub_node.lineno,
                                            "end_line": getattr(
                                                sub_node, "end_lineno", sub_node.lineno
                                            ),
                                            "chunk_start_line": win["chunk_start_line"],
                                            "chunk_end_line": win["chunk_end_line"],
                                            "chunk_index": idx,
                                            "chunk_count": total_method_chunks,
                                            "token_count": win["token_count"],
                                            "embedding_model": self.embedding_model,
                                            "chunking_strategy": "HierarchicalAST",
                                        },
                                    }
                                )

                # Capture global runtime statements, constants, and root initialization imports
                else:
                    element_code = self._get_node_source(source_code, node)
                    if element_code.strip():
                        if global_start_line is None:
                            global_start_line = getattr(node, "lineno", 1)
                        global_elements.append(element_code)

            except Exception as e:
                logger.error(
                    f"Error processing specific node in {file_path} near line {getattr(node, 'lineno', 'unknown')}: {e}"
                )
                continue

        # MODULE GLOBALS ASSEMBLY PASS
        if global_elements:
            combined_globals = "\n".join(global_elements)
            g_start = global_start_line if global_start_line else 1
            windows = self._split_by_token_window(
                combined_globals, node_start_line=g_start
            )
            total_global_chunks = len(windows)

            for idx, win in enumerate(windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": {
                            "repo_name": repo_name,
                            "file_path": rel_path,
                            "filename": path_obj.name,
                            "module_name": path_obj.stem,
                            "language": "python",
                            "extension": ".py",
                            "node_type": "module_globals",
                            "chunk_type": "code_ast",
                            "name": "globals",
                            "parent": None,
                            "parent_type": None,
                            "start_line": g_start,
                            "end_line": len(source_code.splitlines()),
                            "chunk_start_line": win["chunk_start_line"],
                            "chunk_end_line": win["chunk_end_line"],
                            "chunk_index": idx,
                            "chunk_count": total_global_chunks,
                            "token_count": win["token_count"],
                            "embedding_model": self.embedding_model,
                            "chunking_strategy": "HierarchicalAST",
                        },
                    }
                )

        # UPDATE STATE MANAGEMENT SUCCESS LEDGER
        if state_manager is not None and chunks:
            state_manager.mark_as_processed(file_path, repo_root, repo_name)
            logger.info(f"Successfully recorded processed hash state for: {rel_path}")

        return chunks


if __name__ == "__main__":
    logger.info("Running AST Chunker sanity check...")

    chunker = PythonASTChunker(max_tokens=512, overlap_percent=0.15)

    sample_file = os.path.join(
        "data",
        "fastapi",
        "fastapi",
        "utils.py",
    )

    if not os.path.exists(sample_file):
        sample_file = __file__

    # logger.info(f"Target file: {sample_file}")
    # file_chunks = chunker.chunk_file(sample_file)
    # logger.info(f"Total Chunks Extracted: {len(file_chunks)}")

    print(f"Target file: {sample_file}")

    file_chunks = chunker.chunk_file(sample_file)

    print(f"\nTotal Chunks: {len(file_chunks)}")

    for i, chunk in enumerate(file_chunks):
        print(f"\n{'=' * 60}")
        print(f"Chunk {i}")
        print(f"{'=' * 60}")
        print("Metadata:")
        for k, v in chunk["metadata"].items():
            print(f"  {k}: {v}")
        token_count = len(chunker.tokenizer.encode(chunk["text"]))
        print(f"Token Count: {token_count}")
        print("\nPreview:")
        print(chunk["text"])
