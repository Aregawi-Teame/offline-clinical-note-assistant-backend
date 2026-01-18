"""
SQLite audit logger for request/response tracking.
"""
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.generate import TaskEnum

logger = get_logger(__name__)


class AuditLogger:
    """Audit logger for storing request metadata in SQLite."""
    
    def __init__(self):
        """Initialize audit logger."""
        self.enabled = settings.STORE_REQUESTS
        self.db_path = settings.SQLITE_PATH
        self._connection: Optional[sqlite3.Connection] = None
        
        if self.enabled:
            self._ensure_db_directory()
            self._initialize_database()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        db_dir = self.db_path.parent
        if db_dir and not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
    
    def _initialize_database(self):
        """Initialize SQLite database and create table if needed."""
        try:
            self._connection = sqlite3.connect(str(self.db_path))
            self._connection.row_factory = sqlite3.Row
            
            # Create table if it doesn't exist
            cursor = self._connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    notes_hash TEXT NOT NULL,
                    output_hash TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(request_id)
                )
            """)
            
            # Create index on request_id for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_id ON audit_log(request_id)
            """)
            
            # Create index on timestamp for time-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)
            """)
            
            self._connection.commit()
            logger.info(f"Audit logger initialized. Database: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
            self.enabled = False
            if self._connection:
                self._connection.close()
                self._connection = None
    
    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Get database connection, reconnecting if needed."""
        if not self.enabled:
            return None
        
        try:
            if self._connection is None:
                self._connection = sqlite3.connect(str(self.db_path))
                self._connection.row_factory = sqlite3.Row
            return self._connection
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            return None
    
    def _compute_hash(self, data: str) -> str:
        """
        Compute SHA256 hash of data.
        
        Args:
            data: String data to hash
            
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def log(
        self,
        request_id: str,
        task: TaskEnum,
        notes: str,
        output: str,
        latency_ms: float
    ) -> bool:
        """
        Log request/response metadata to SQLite database.
        
        Args:
            request_id: Unique request identifier
            task: Task type (SOAP, DISCHARGE, REFERRAL)
            notes: Input notes (will be hashed, not stored)
            output: Generated output (will be hashed, not stored)
            latency_ms: Request latency in milliseconds
            
        Returns:
            True if logged successfully, False otherwise
        """
        if not self.enabled:
            return False
        
        try:
            conn = self._get_connection()
            if conn is None:
                return False
            
            # Compute hashes
            notes_hash = self._compute_hash(notes)
            output_hash = self._compute_hash(output)
            
            # Get timestamp
            timestamp = datetime.utcnow().isoformat()
            
            # Insert record
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO audit_log 
                (request_id, timestamp, task, notes_hash, output_hash, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (request_id, timestamp, task.value, notes_hash, output_hash, latency_ms))
            
            conn.commit()
            logger.debug(f"Audit log recorded for request: {request_id}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Failed to log audit entry: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in audit logging: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
