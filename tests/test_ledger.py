import os
from src.db.state_manager import IngestionStateManager

# Setup paths
repo_root = os.path.abspath("./data")

file_a = os.path.join(repo_root, "fastapi", "fastapi", "utils.py")
file_b = os.path.join(
    repo_root, "qdrant-client", "qdrant_client", "common", "version_check.py"
)

# Initialize Ledger
ledger = IngestionStateManager("sqlite:///data/test_ledger.db")


# Test Ingestion
def test_file(abs_path, repo):
    print(f"\nTesting {repo} -> {os.path.basename(abs_path)}...")
    if ledger.needs_processing(abs_path, repo_root, repo):
        # Pretend we processed it
        ledger.mark_as_processed(abs_path, repo_root, repo)
        print("-> Ingested.")
    else:
        print("-> Skipped.")


test_file(file_a, "repoA")  # Should be NEW
test_file(file_a, "repoA")  # Should be SKIPPED
test_file(file_b, "repoB")  # Should be NEW (different repo, same filename!)
test_file(file_b, "repoB")  # Should be SKIPPED
