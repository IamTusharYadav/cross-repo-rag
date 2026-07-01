import ast
import os
import re
import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import tiktoken
from src.db.state_manager import IngestionStateManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
)
logger = logging.getLogger(__name__)


class CallVisitor(ast.NodeVisitor):
    """
    AST Walker to extract direct function calls and external symbol references
    from within a function or method body for Call Graph Metadata.
    """

    def __init__(self):
        self.calls = set()
        self.external_symbols = set()

    def visit_Call(self, node):
        try:
            if isinstance(node.func, ast.Name):
                self.calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                self.calls.add(node.func.attr)  # Extract the method name being called

                # Extract the base object/module being called (e.g., qdrant_client in qdrant_client.upsert)
                base_val = ast.unparse(node.func.value)
                # Only record simple identifiers, ignoring complex chained calls
                if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", base_val):
                    self.external_symbols.add(base_val)
        except Exception:
            pass
        self.generic_visit(node)


class PythonASTChunker:
    """
    An Abstract Syntax Tree (AST) chunker that parses Python source files, isolates meaningful logical code blocks
    (modules, functions, classes, methods), tracks hierarchical parent-child relationships and maps precise line
    boundaries for accurate chunk-level source highlighting.

    Production Improvements (v2 Cross-Repo Update):
    - Injects class signatures directly into method chunks for deeper semantic context.
    - Implements recursive DFS extraction for nested inner classes.
    - Captures Async/Await metadata boolean flags.
    - Resolves Base Class inheritance arrays.
    - Call Graph analysis (function calls + external module references).
    - Memoized token cache optimization for massive throughput gains.
    - Distinct Module-Level Constants separation.
    - Deterministic SHA-256 chunk UUID tracking.
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
        self._token_cache: Dict[str, int] = {}

        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.warning(
                "cl100k_base encoding not found, falling back to gpt-4 tokenizer."
            )
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")

    # UTILITIES & SOURCE EXTRACTION
    def _count_tokens(self, text: str) -> int:
        """Cached token counting to prevent redundant O(N) operations on massive files."""
        if text not in self._token_cache:
            self._token_cache[text] = len(self.tokenizer.encode(text))
        return self._token_cache[text]

    def _generate_chunk_id(
        self, symbol_path: str, start_line: int, end_line: int, text: str
    ) -> str:
        """Deterministic UUID tracking for eval, deduplication, and caching."""
        raw = f"{symbol_path}_{start_line}_{end_line}_{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_node_source(self, source_code: str, node: ast.AST) -> str:
        try:
            segment = ast.get_source_segment(source_code, node)
            if segment:
                return segment
        except Exception:
            pass

        lines = source_code.splitlines()
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", len(lines))
        return "\n".join(lines[start_line:end_line])

    def _get_class_signature_chunk(self, source_code: str, node: ast.ClassDef) -> str:
        lines = source_code.splitlines()
        deco_start = node.lineno - 1
        if node.decorator_list:
            deco_start = node.decorator_list[0].lineno - 1

        class_line_idx = node.lineno - 1
        header_lines = lines[deco_start : class_line_idx + 1]

        docstring = ast.get_docstring(node)
        if docstring and node.body:
            first_child = node.body[0]
            if isinstance(first_child, ast.Expr) and isinstance(
                first_child.value, (ast.Constant, ast.Str)
            ):
                doc_end = getattr(first_child, "end_lineno", class_line_idx + 1)
                header_lines += lines[class_line_idx + 1 : doc_end]

        return "\n".join(header_lines)

    def _get_function_signature(self, source_code: str, node: ast.AST) -> str:
        lines = source_code.splitlines()
        deco_start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            deco_start = node.decorator_list[0].lineno - 1

        func_line_idx = node.lineno - 1
        header_lines = lines[deco_start : func_line_idx + 1]

        docstring = ast.get_docstring(node)
        if docstring and getattr(node, "body", None):
            first_child = node.body[0]
            if isinstance(first_child, ast.Expr) and isinstance(
                first_child.value, (ast.Constant, ast.Str)
            ):
                doc_end = getattr(first_child, "end_lineno", func_line_idx + 1)
                header_lines += lines[func_line_idx + 1 : doc_end]

        return "\n".join(header_lines)

    def _extract_decorator_names(self, node: ast.AST) -> List[str]:
        decorators = []
        for deco in getattr(node, "decorator_list", []):
            if isinstance(deco, ast.Name):
                decorators.append(deco.id)
            elif isinstance(deco, ast.Attribute):
                decorators.append(f"{ast.unparse(deco)}")
            elif isinstance(deco, ast.Call):
                func = deco.func
                if isinstance(func, ast.Name):
                    decorators.append(f"{func.id}(...)")
                elif isinstance(func, ast.Attribute):
                    decorators.append(f"{ast.unparse(func)}(...)")
        return decorators

    def _extract_imports(self, tree: ast.Module) -> List[str]:
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        f"import {alias.name}"
                        + (f" as {alias.asname}" if alias.asname else "")
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(
                    (a.asname if a.asname else a.name) for a in node.names
                )
                imports.append(f"from {module} import {names}")
        return imports

    def _is_constant_assign(self, node: ast.AST) -> bool:
        """Detect module-level ALL_CAPS constants."""
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    return True
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id.isupper():
                return True
        return False

    # TOKEN HELPERS & SPLITTING LOGIC

    def _split_by_line_window(
        self, text: str, node_start_line: int, context_header: str = ""
    ) -> List[Dict[str, Any]]:
        header_tokens = self._count_tokens(context_header) if context_header else 0
        usable_tokens = max(60, self.max_tokens - header_tokens)
        raw_lines = text.splitlines(keepends=True)

        if not raw_lines:
            return []

        if self._count_tokens(text) <= usable_tokens:
            full_text = context_header + text
            return [
                {
                    "text": full_text,
                    "chunk_start_line": node_start_line,
                    "chunk_end_line": node_start_line + max(0, len(raw_lines) - 1),
                    "token_count": self._count_tokens(full_text),
                }
            ]

        sub_chunks: List[Dict[str, Any]] = []
        current_lines: List[str] = []
        current_tokens = 0
        line_idx = 0
        window_start_idx = 0

        while line_idx < len(raw_lines):
            line = raw_lines[line_idx]
            line_tokens = self._count_tokens(line)

            if line_tokens > usable_tokens:
                if current_lines:
                    payload = context_header + "".join(current_lines)
                    sub_chunks.append(
                        {
                            "text": payload,
                            "chunk_start_line": node_start_line + window_start_idx,
                            "chunk_end_line": node_start_line + line_idx - 1,
                            "token_count": self._count_tokens(payload),
                        }
                    )
                    current_lines, current_tokens = [], 0
                    window_start_idx = line_idx

                payload = context_header + line
                sub_chunks.append(
                    {
                        "text": payload,
                        "chunk_start_line": node_start_line + line_idx,
                        "chunk_end_line": node_start_line + line_idx,
                        "token_count": self._count_tokens(payload),
                    }
                )
                line_idx += 1
                window_start_idx = line_idx
                continue

            if current_tokens + line_tokens > usable_tokens:
                payload = context_header + "".join(current_lines)
                sub_chunks.append(
                    {
                        "text": payload,
                        "chunk_start_line": node_start_line + window_start_idx,
                        "chunk_end_line": node_start_line + line_idx - 1,
                        "token_count": self._count_tokens(payload),
                    }
                )

                overlap_lines: List[str] = []
                overlap_tokens = 0
                for back_line in reversed(current_lines):
                    bt = self._count_tokens(back_line)
                    if overlap_tokens + bt > self.overlap_tokens:
                        break
                    overlap_lines.insert(0, back_line)
                    overlap_tokens += bt

                backtrack_count = len(overlap_lines)
                window_start_idx = line_idx - backtrack_count
                current_lines = overlap_lines
                current_tokens = overlap_tokens

            current_lines.append(line)
            current_tokens += line_tokens
            line_idx += 1

        if current_lines:
            payload = context_header + "".join(current_lines)
            sub_chunks.append(
                {
                    "text": payload,
                    "chunk_start_line": node_start_line + window_start_idx,
                    "chunk_end_line": node_start_line + len(raw_lines) - 1,
                    "token_count": self._count_tokens(payload),
                }
            )

        return sub_chunks

    def _split_function_by_chunks(
        self,
        source_code: str,
        node: ast.AST,
        module_name: str,
        class_name: Optional[str],
        parent_class_sig: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """AST-Aware semantic statement splitting with Class-Signature Injection."""
        func_code = self._get_node_source(source_code, node)
        node_type = "method" if class_name else "function"

        breadcrumb_header = self._build_breadcrumb(
            module_name, class_name, node.name, node_type
        )

        # Inject Class Context for reranker visibility
        if parent_class_sig:
            breadcrumb_header += f"Class Signature Context:\n{parent_class_sig}\n\n"

        deco_start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            deco_start = node.decorator_list[0].lineno - 1
        func_start_line = deco_start + 1

        total_tokens = self._count_tokens(breadcrumb_header + func_code)
        if total_tokens <= self.max_tokens:
            return [
                {
                    "text": breadcrumb_header + func_code,
                    "chunk_start_line": func_start_line,
                    "chunk_end_line": getattr(node, "end_lineno", node.lineno),
                    "token_count": total_tokens,
                }
            ]

        sig_code = self._get_function_signature(source_code, node)
        overhead_tokens = self._count_tokens(breadcrumb_header + sig_code + "\n")
        usable_tokens = self.max_tokens - overhead_tokens

        if usable_tokens < 100:
            return self._split_by_line_window(
                func_code,
                node_start_line=func_start_line,
                context_header=breadcrumb_header,
            )

        sub_chunks: List[Dict[str, Any]] = []
        current_group: List[Dict[str, Any]] = []
        current_tokens = 0

        statements = node.body
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, (ast.Constant, ast.Str))
        ):
            statements = statements[1:]

        if not statements:
            payload = breadcrumb_header + sig_code
            return [
                {
                    "text": payload,
                    "chunk_start_line": func_start_line,
                    "chunk_end_line": getattr(node, "end_lineno", node.lineno),
                    "token_count": self._count_tokens(payload),
                }
            ]

        idx = 0
        while idx < len(statements):
            stmt = statements[idx]
            stmt_start = stmt.lineno
            stmt_end = getattr(stmt, "end_lineno", stmt_start)
            stmt_code = self._get_node_source(source_code, stmt)
            stmt_tokens = self._count_tokens(stmt_code)

            if stmt_tokens > usable_tokens:
                if current_group:
                    payload = (
                        breadcrumb_header
                        + sig_code
                        + "\n"
                        + "\n".join([g["code"] for g in current_group])
                    )
                    sub_chunks.append(
                        {
                            "text": payload,
                            "chunk_start_line": current_group[0]["start"],
                            "chunk_end_line": current_group[-1]["end"],
                            "token_count": self._count_tokens(payload),
                        }
                    )
                    current_group, current_tokens = [], 0

                fallback_header = breadcrumb_header + sig_code + "\n"
                stmt_windows = self._split_by_line_window(
                    stmt_code,
                    node_start_line=stmt_start,
                    context_header=fallback_header,
                )
                sub_chunks.extend(stmt_windows)
                idx += 1
                continue

            if current_tokens + stmt_tokens > usable_tokens:
                payload = (
                    breadcrumb_header
                    + sig_code
                    + "\n"
                    + "\n".join([g["code"] for g in current_group])
                )
                sub_chunks.append(
                    {
                        "text": payload,
                        "chunk_start_line": current_group[0]["start"],
                        "chunk_end_line": current_group[-1]["end"],
                        "token_count": self._count_tokens(payload),
                    }
                )

                overlap_group = []
                overlap_tokens = 0
                for back_stmt in reversed(current_group):
                    if overlap_tokens + back_stmt["tokens"] > self.overlap_tokens:
                        break
                    overlap_group.insert(0, back_stmt)
                    overlap_tokens += back_stmt["tokens"]

                current_group = overlap_group
                current_tokens = overlap_tokens

            current_group.append(
                {
                    "code": stmt_code,
                    "tokens": stmt_tokens,
                    "start": stmt_start,
                    "end": stmt_end,
                }
            )
            current_tokens += stmt_tokens
            idx += 1

        if current_group:
            payload = (
                breadcrumb_header
                + sig_code
                + "\n"
                + "\n".join([g["code"] for g in current_group])
            )
            sub_chunks.append(
                {
                    "text": payload,
                    "chunk_start_line": current_group[0]["start"],
                    "chunk_end_line": current_group[-1]["end"],
                    "token_count": self._count_tokens(payload),
                }
            )

        return sub_chunks

    # REPO / PATH HELPERS

    def _extract_repo_name(self, file_path: str) -> str:
        normalized_path = os.path.normpath(file_path)
        parts_path = normalized_path.split(os.sep)
        if "data" in parts_path:
            data_index = parts_path.index("data")
            if data_index + 1 < len(parts_path):
                return parts_path[data_index + 1]
        return "unknown"

    def _build_symbol_path(
        self,
        repo_name: str,
        module_name: str,
        class_name: Optional[str],
        node_name: str,
        node_type: str = "function",
    ) -> str:
        """Explicit Method Identifiers (::)"""
        parts = [repo_name, module_name]
        if class_name:
            parts.append(class_name)
        base = ".".join(parts)

        if node_type == "method" or node_type == "constant":
            return f"{base}::{node_name}"
        return f"{base}.{node_name}"

    def _build_breadcrumb_metadata(
        self, module_name: str, class_name: Optional[str], node_name: Optional[str]
    ) -> str:
        parts = [module_name]
        if class_name:
            parts.append(class_name)
        if node_name:
            parts.append(node_name)
        return " > ".join(parts)

    def _build_breadcrumb(
        self,
        module_name: str,
        class_name: Optional[str],
        node_name: Optional[str],
        node_type: str,
    ) -> str:
        lines = [f"Module: {module_name}"]
        if class_name:
            lines.append(f"Class: {class_name}")
        if node_name and node_type not in ("module_docstring", "module_globals"):
            label = "Method" if node_type == "method" else node_type.capitalize()
            lines.append(f"{label}: {node_name}")
        lines.append("-" * 40)
        return "\n".join(lines) + "\n\n"

    # MAIN ENTRY & DFS PARSING ENGINE

    def chunk_file(
        self,
        file_path: str,
        state_manager: IngestionStateManager = None,
        repo_root: str = "./data",
    ) -> List[Dict[str, Any]]:

        chunks: List[Dict[str, Any]] = []
        if not os.path.isfile(file_path):
            logger.error(f"File target structural absence flagged: {file_path}")
            return chunks

        repo_name = self._extract_repo_name(file_path)
        rel_path = os.path.relpath(file_path)
        path_obj = Path(rel_path)

        normalized_path = os.path.normpath(file_path)
        path_parts = normalized_path.split(os.sep)
        filename = path_parts[-1] if path_parts else ""
        dir_parts = set(path_parts[:-1])

        # ignore lists for FastAPI + Qdrant
        ignored_dirs = {
            "tests",
            "__pycache__",
            "htmlcov",
            ".venv",
            ".env",
            "scripts",
            "tools",
            ".github",
            "benchmarks",
            ".agents",
        }
        ignored_files = {".env", "conftest.py"}

        # Guard both folder segments and specific standalone filenames
        if dir_parts.intersection(ignored_dirs) or filename in ignored_files:
            logger.info(f"Skipping ignored file or directory path: {normalized_path}")
            return chunks

        if state_manager is not None:
            if not state_manager.needs_processing(file_path, repo_root, repo_name):
                logger.info(
                    f"File cache hit. Skipping structural parsing for: {rel_path}"
                )
                return chunks

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
            tree = ast.parse(source_code, filename=file_path)
        except Exception as e:
            logger.warning(f"Error accessing or parsing {file_path}: {e}")
            return chunks

        module_name = path_obj.stem
        import_context: List[str] = self._extract_imports(tree)

        def base_meta(**overrides) -> Dict[str, Any]:
            m = {
                "repo_name": repo_name,
                "file_path": rel_path,
                "filename": path_obj.name,
                "module_name": module_name,
                "language": "python",
                "extension": ".py",
                "embedding_model": self.embedding_model,
                "chunking_strategy": "HierarchicalAST",
                "import_context": import_context,
            }
            m.update(overrides)
            return m

        # DOCSTRING Extraction
        try:
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                doc_text = f'"""\n{module_docstring}\n"""'
                breadcrumb_txt = self._build_breadcrumb(
                    module_name, None, None, "module_docstring"
                )
                breadcrumb_meta = self._build_breadcrumb_metadata(
                    module_name, None, "docstring"
                )
                windows = self._split_by_line_window(
                    doc_text, node_start_line=1, context_header=breadcrumb_txt
                )

                symbol_path = f"{repo_name}.{module_name}"
                for idx, win in enumerate(windows):
                    chunk_id = self._generate_chunk_id(
                        symbol_path,
                        win["chunk_start_line"],
                        win["chunk_end_line"],
                        win["text"],
                    )
                    chunks.append(
                        {
                            "text": win["text"],
                            "metadata": base_meta(
                                chunk_id=chunk_id,
                                node_type="module_docstring",
                                chunk_type="docstring",
                                name=module_name,
                                parent=None,
                                parent_type=None,
                                symbol_path=symbol_path,
                                breadcrumb=breadcrumb_meta,
                                decorators=[],
                                start_line=1,
                                end_line=len(doc_text.splitlines()),
                                chunk_start_line=win["chunk_start_line"],
                                chunk_end_line=win["chunk_end_line"],
                                chunk_index=idx,
                                chunk_count=len(windows),
                                token_count=win["token_count"],
                            ),
                        }
                    )
        except Exception:
            pass

        # DFS Traversal Functions
        def process_function(
            func_node: ast.AST, parent_class_name=None, parent_class_sig=None
        ):
            is_async = isinstance(func_node, ast.AsyncFunctionDef)  # Async
            decorators = self._extract_decorator_names(func_node)

            # Call graph & External modules
            visitor = CallVisitor()
            visitor.visit(func_node)
            calls = list(visitor.calls)
            external_symbols = list(visitor.external_symbols)

            node_type = "method" if parent_class_name else "function"
            symbol_name = (
                f"{parent_class_name}::{func_node.name}"
                if parent_class_name
                else func_node.name
            )

            symbol_path = self._build_symbol_path(
                repo_name, module_name, parent_class_name, func_node.name, node_type
            )
            breadcrumb_meta = self._build_breadcrumb_metadata(
                module_name, parent_class_name, func_node.name
            )

            windows = self._split_function_by_chunks(
                source_code, func_node, module_name, parent_class_name, parent_class_sig
            )
            for idx, win in enumerate(windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": base_meta(
                            chunk_id=self._generate_chunk_id(
                                symbol_path,
                                win["chunk_start_line"],
                                win["chunk_end_line"],
                                win["text"],
                            ),
                            node_type=node_type,
                            chunk_type="code_ast",
                            name=symbol_name,
                            parent=parent_class_name,
                            parent_type="class" if parent_class_name else None,
                            symbol_path=symbol_path,
                            breadcrumb=breadcrumb_meta,
                            decorators=decorators,
                            is_async=is_async,
                            calls=calls,
                            external_symbols=external_symbols,
                            start_line=func_node.lineno,
                            end_line=getattr(func_node, "end_lineno", func_node.lineno),
                            chunk_start_line=win["chunk_start_line"],
                            chunk_end_line=win["chunk_end_line"],
                            chunk_index=idx,
                            chunk_count=len(windows),
                            token_count=win["token_count"],
                        ),
                    }
                )

        def process_class(class_node: ast.ClassDef, parent_class_name=None):
            class_name = (
                f"{parent_class_name}.{class_node.name}"
                if parent_class_name
                else class_node.name
            )
            decorators = self._extract_decorator_names(class_node)

            # Base classes
            base_classes = []
            for b in getattr(class_node, "bases", []):
                try:
                    base_classes.append(ast.unparse(b))
                except Exception:
                    pass

            symbol_path = self._build_symbol_path(
                repo_name, module_name, parent_class_name, class_node.name, "class"
            )
            breadcrumb_meta = self._build_breadcrumb_metadata(
                module_name, parent_class_name, class_node.name
            )

            deco_start = (
                class_node.decorator_list[0].lineno - 1
                if class_node.decorator_list
                else class_node.lineno - 1
            )
            class_start_line = deco_start + 1

            class_header_code = self._get_class_signature_chunk(source_code, class_node)
            class_breadcrumb_txt = self._build_breadcrumb(
                module_name, parent_class_name, class_node.name, "class"
            )

            class_windows = self._split_by_line_window(
                class_header_code,
                node_start_line=class_start_line,
                context_header=class_breadcrumb_txt,
            )

            for idx, win in enumerate(class_windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": base_meta(
                            chunk_id=self._generate_chunk_id(
                                symbol_path,
                                win["chunk_start_line"],
                                win["chunk_end_line"],
                                win["text"],
                            ),
                            node_type="class",
                            chunk_type="code_ast",
                            name=class_name,
                            parent=parent_class_name,
                            parent_type="class" if parent_class_name else None,
                            symbol_path=symbol_path,
                            breadcrumb=breadcrumb_meta,
                            decorators=decorators,
                            base_classes=base_classes,
                            start_line=class_node.lineno,
                            end_line=getattr(
                                class_node, "end_lineno", class_node.lineno
                            ),
                            chunk_start_line=win["chunk_start_line"],
                            chunk_end_line=win["chunk_end_line"],
                            chunk_index=idx,
                            chunk_count=len(class_windows),
                            token_count=win["token_count"],
                        ),
                    }
                )

            # Recursive DFS for nested classes and methods
            for sub_node in class_node.body:
                if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(
                        sub_node,
                        parent_class_name=class_name,
                        parent_class_sig=class_header_code,
                    )
                elif isinstance(sub_node, ast.ClassDef):
                    process_class(sub_node, parent_class_name=class_name)

        def process_constant(node: ast.AST):
            """Dedicated module-level constant abstraction."""
            const_code = self._get_node_source(source_code, node)
            if not const_code.strip():
                return

            start_line = getattr(node, "lineno", 1)
            end_line = getattr(node, "end_lineno", start_line)

            names = []
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.append(node.target.id)

            name_str = ", ".join(names) if names else "Constant"
            symbol_path = self._build_symbol_path(
                repo_name, module_name, None, name_str, "constant"
            )
            breadcrumb_txt = (
                f"Module: {module_name}\nConstant: {name_str}\n" + ("-" * 40) + "\n\n"
            )

            windows = self._split_by_line_window(const_code, start_line, breadcrumb_txt)
            for idx, win in enumerate(windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": base_meta(
                            chunk_id=self._generate_chunk_id(
                                symbol_path,
                                win["chunk_start_line"],
                                win["chunk_end_line"],
                                win["text"],
                            ),
                            node_type="constant",
                            chunk_type="code_ast",
                            name=name_str,
                            parent=None,
                            parent_type=None,
                            symbol_path=symbol_path,
                            breadcrumb=self._build_breadcrumb_metadata(
                                module_name, None, name_str
                            ),
                            start_line=start_line,
                            end_line=end_line,
                            chunk_start_line=win["chunk_start_line"],
                            chunk_end_line=win["chunk_end_line"],
                            chunk_index=idx,
                            chunk_count=len(windows),
                            token_count=win["token_count"],
                        ),
                    }
                )

        # Core Module Execution Route
        global_elements: List[str] = []
        global_start_line: Optional[int] = None

        for node in tree.body:
            try:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    process_function(node)
                elif isinstance(node, ast.ClassDef):
                    process_class(node)
                elif self._is_constant_assign(node):
                    process_constant(node)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                else:
                    element_code = self._get_node_source(source_code, node)
                    if element_code.strip():
                        if global_start_line is None:
                            global_start_line = getattr(node, "lineno", 1)
                        global_elements.append(element_code)
            except Exception as e:
                logger.error(
                    f"Error processing node in {file_path} near line {getattr(node, 'lineno', 'unknown')}: {e}"
                )

        # MODULE GLOBALS ASSEMBLY
        if global_elements:
            combined_globals = "\n".join(global_elements)
            g_start = global_start_line if global_start_line else 1
            globals_breadcrumb_txt = self._build_breadcrumb(
                module_name, None, None, "module_globals"
            )
            globals_breadcrumb_meta = self._build_breadcrumb_metadata(
                module_name, None, "globals"
            )
            windows = self._split_by_line_window(
                combined_globals,
                node_start_line=g_start,
                context_header=globals_breadcrumb_txt,
            )

            global_symbol = f"{repo_name}.{module_name}.globals"
            for idx, win in enumerate(windows):
                chunks.append(
                    {
                        "text": win["text"],
                        "metadata": base_meta(
                            chunk_id=self._generate_chunk_id(
                                global_symbol,
                                win["chunk_start_line"],
                                win["chunk_end_line"],
                                win["text"],
                            ),
                            node_type="module_globals",
                            chunk_type="code_ast",
                            name="globals",
                            parent=None,
                            parent_type=None,
                            symbol_path=global_symbol,
                            breadcrumb=globals_breadcrumb_meta,
                            decorators=[],
                            start_line=g_start,
                            end_line=len(source_code.splitlines()),
                            chunk_start_line=win["chunk_start_line"],
                            chunk_end_line=win["chunk_end_line"],
                            chunk_index=idx,
                            chunk_count=len(windows),
                            token_count=win["token_count"],
                        ),
                    }
                )

        if state_manager is not None and chunks:
            state_manager.mark_as_processed(file_path, repo_root, repo_name)
            logger.info(f"Successfully recorded processed hash state for: {rel_path}")

        return chunks


if __name__ == "__main__":
    logger.info("Running AST Chunker v4 Cross-Repo Update sanity check...")
    chunker = PythonASTChunker(max_tokens=512, overlap_percent=0.15)

    sample_file = os.path.join("data", "fastapi", "fastapi", "utils.py")
    if not os.path.exists(sample_file):
        sample_file = __file__

    print(f"Target file: {sample_file}")
    file_chunks = chunker.chunk_file(sample_file)
    print(f"\nTotal Chunks: {len(file_chunks)}")

    for i, chunk in enumerate(
        file_chunks[:5]
    ):  # Printing just first 5 to prevent console flood
        print(f"\n{'=' * 60}\nChunk {i}\n{'=' * 60}")
        print("Metadata:")
        for k, v in chunk["metadata"].items():
            print(f"  {k}: {v}")
        token_count = len(chunker.tokenizer.encode(chunk["text"]))
        print(f"Token Count: {token_count}")
        print("\nPreview:")
        print(chunk["text"])
