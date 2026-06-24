import hashlib
import os
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s",
)
logger = logging.getLogger(__name__)

Base = declarative_base()


class FileState(Base):
    """SQLAlchemy Model for tracking file ingestion state"""

    __tablename__ = "ingestion_ledger"

    # Composite Primary Key
    repo_name = Column(String, primary_key=True)
    file_path = Column(String, primary_key=True)

    # Metadata & State
    absolute_path = Column(String, nullable=False)
    content_hash = Column(String, nullable=False)
    last_modified = Column(Float, nullable=False)
    indexed_at = Column(DateTime, nullable=False)


class IngestionStateManager:
    """
    Manages the SQLite database that tracks file hashes.
    """

    def __init__(self, db_path: str = "sqlite:///data/ingestion_state.db"):
        os.makedirs(os.path.dirname(db_path.replace("sqlite:///", "")), exist_ok=True)
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def _compute_sha256(self, absolute_path: str) -> str:
        """Computes the SHA-256 hash of a file in 8KB chunks."""
        hasher = hashlib.sha256()
        with open(absolute_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def needs_processing(
        self, absolute_path: str, repo_root: str, repo_name: str
    ) -> bool:
        """
        Checks if a file needs ingestion using its relative path for lookup.
        """
        if not os.path.exists(absolute_path):
            logger.error(f"Cannot check state for missing file: {absolute_path}")
            return False

        relative_path = os.path.relpath(absolute_path, repo_root).replace("\\", "/")
        current_hash = self._compute_sha256(absolute_path)

        with self.Session() as session:
            record = (
                session.query(FileState)
                .filter_by(repo_name=repo_name, file_path=relative_path)
                .first()
            )

            if record:
                if record.content_hash == current_hash:
                    logger.info(f"SKIPPED: Unchanged file {relative_path}")
                    return False
                else:
                    logger.info(f"UPDATE DETECTED: File modified {relative_path}")
                    return True

            logger.info(f"NEW FILE: {relative_path}")
            return True

    def mark_as_processed(self, absolute_path: str, repo_root: str, repo_name: str):
        """
        Records successful ingestion.
        """
        relative_path = os.path.relpath(absolute_path, repo_root).replace("\\", "/")
        current_hash = self._compute_sha256(absolute_path)
        current_mtime = os.path.getmtime(absolute_path)
        now = datetime.now(timezone.utc)

        with self.Session() as session:
            record = (
                session.query(FileState)
                .filter_by(repo_name=repo_name, file_path=relative_path)
                .first()
            )

            if record:
                record.content_hash = current_hash
                record.last_modified = current_mtime
                record.indexed_at = now
                record.absolute_path = (
                    absolute_path  # Update just in case the clone moved
                )
            else:
                new_record = FileState(
                    repo_name=repo_name,
                    file_path=relative_path,
                    absolute_path=absolute_path,
                    content_hash=current_hash,
                    last_modified=current_mtime,
                    indexed_at=now,
                )
                session.add(new_record)

            session.commit()
            logger.info(f"RECORDED: {relative_path} secured in ledger.")

    def get_last_indexed_time(
        self, absolute_path: str, repo_root: str, repo_name: str
    ) -> str:
        """Retrieves the indexed timestamp using the composite key."""
        relative_path = os.path.relpath(absolute_path, repo_root).replace("\\", "/")
        with self.Session() as session:
            record = (
                session.query(FileState)
                .filter_by(repo_name=repo_name, file_path=relative_path)
                .first()
            )
            if record:
                return record.indexed_at.isoformat()
        return datetime.now(timezone.utc).isoformat()

    def get_all_tracked_files(self) -> list[dict]:
        """Returns a safe dictionary representation of all tracked files to prevent ORM leakage."""
        with self.Session() as session:
            records = session.query(FileState).all()
            return [
                {
                    "repo_name": r.repo_name,
                    "file_path": r.file_path,
                    "absolute_path": r.absolute_path,
                }
                for r in records
            ]

    def remove_file_record(self, repo_name: str, file_path: str):
        """Drops a specific file from the ledger (used during reconciliation)."""
        with self.Session() as session:
            session.query(FileState).filter_by(
                repo_name=repo_name, file_path=file_path
            ).delete()
            session.commit()
            logger.info(f"LEDGER PURGED: {file_path}")
