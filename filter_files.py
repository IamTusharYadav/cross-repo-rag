import os

DATA_DIR = "data"
ALLOWED_EXTENSIONS = {".py", ".md"}


def should_keep_file(repo_name, root, file):
    _, ext = os.path.splitext(file.lower())
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Prune non-English documentation to keep context lean
    if repo_name == "fastapi" and "docs" in root:
        docs_root = os.path.join(DATA_DIR, "fastapi", "docs")
        relative_from_docs = os.path.relpath(root, docs_root)

        if relative_from_docs != ".":
            first_sub_dir = relative_from_docs.split(os.sep)[0]
            if first_sub_dir != "en":
                return False

    return True


def filter_repository(repo_name, repo_path):
    kept_count = 0
    deleted_count = 0

    # Walk and delete non-target assets
    for root, dirs, files in os.walk(repo_path, topdown=True):
        # Instantly drop hidden directories, test suites, and redundant source snippets.
        # Modifying dirs[:] in-place prevents os.walk from even entering these heavy directories.
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".")
            and d.lower() not in {"tests", "docs_src", "htmlcov", "__pycache__"}
        ]

        for file in files:
            file_path = os.path.join(root, file)

            if should_keep_file(repo_name, root, file):
                kept_count += 1
            else:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Could not remove {file_path}: {e}")

    # Clean up empty directories left behind (bottom-up sweep)
    for root, dirs, files in os.walk(repo_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if os.path.exists(dir_path) and not os.listdir(dir_path):
                    os.rmdir(dir_path)
            except Exception:
                pass

    return kept_count, deleted_count


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: '{DATA_DIR}' directory not found.")
        exit(1)

    total_files_kept = 0
    print("Starting repository filtering & noise reduction...")

    for repo_folder in ["fastapi", "qdrant-client"]:
        target_path = os.path.join(DATA_DIR, repo_folder)
        if os.path.exists(target_path):
            kept, deleted = filter_repository(repo_folder, target_path)
            print(f"[{repo_folder}]")
            print(
                f"   Processed: Keep {kept} files | Purged {deleted} non-target assets/tests."
            )
            total_files_kept += kept
        else:
            print(f"Warning: Expected folder structure not found at {target_path}")
