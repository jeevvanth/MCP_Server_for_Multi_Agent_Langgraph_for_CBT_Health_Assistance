
import psycopg2
from psycopg2.extras import Json
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointTuple
from typing import Optional, Dict, Any, Iterator, Sequence
import json
from datetime import datetime
import uuid

class PostgresCheckpointer(BaseCheckpointSaver):
    """PostgreSQL-based checkpointer for LangGraph memory persistence."""
    
    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL checkpointer.
        
        Args:
            connection_string: PostgreSQL connection string
                Example: "postgresql://user:password@localhost:5432/dbname"
        """
        self.connection_string = connection_string
        self._create_tables()
    
    def _get_connection(self):
        """Create a new database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _create_tables(self):
        """Create necessary tables for checkpoint storage."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        parent_checkpoint_id TEXT,
                        type TEXT,
                        checkpoint JSONB NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_thread_id 
                    ON checkpoints(thread_id);
                    
                    CREATE INDEX IF NOT EXISTS idx_parent_id
                    ON checkpoints(parent_checkpoint_id);
                    
                    CREATE TABLE IF NOT EXISTS checkpoint_writes (
                        thread_id TEXT NOT NULL,
                        checkpoint_ns TEXT NOT NULL DEFAULT '',
                        checkpoint_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        idx INTEGER NOT NULL,
                        channel TEXT NOT NULL,
                        value JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                    );
                """)
                conn.commit()
    
    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """
        Retrieve a checkpoint tuple from PostgreSQL.
        
        Args:
            config: Configuration containing thread_id and optionally checkpoint_id
            
        Returns:
            CheckpointTuple or None if not found
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        if not thread_id:
            return None
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if checkpoint_id:
                    # Get specific checkpoint
                    cur.execute("""
                        SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
                    """, (thread_id, checkpoint_ns, checkpoint_id))
                else:
                    # Get latest checkpoint for thread
                    cur.execute("""
                        SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = %s AND checkpoint_ns = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (thread_id, checkpoint_ns))
                
                result = cur.fetchone()
                
                if not result:
                    return None
                
                checkpoint_id, checkpoint_data, metadata, parent_id = result
                
                # Get pending writes for this checkpoint
                cur.execute("""
                    SELECT task_id, channel, value
                    FROM checkpoint_writes
                    WHERE thread_id = %s 
                    AND checkpoint_ns = %s 
                    AND checkpoint_id = %s
                    ORDER BY task_id, idx
                """, (thread_id, checkpoint_ns, checkpoint_id))
                
                pending_writes = []
                for task_id, channel, value in cur.fetchall():
                    pending_writes.append((task_id, channel, value))
                
                parent_config = None
                if parent_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id
                        }
                    }
                
                return CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": checkpoint_id
                        }
                    },
                    checkpoint=checkpoint_data,
                    metadata=metadata or {},
                    parent_config=parent_config,
                    pending_writes=pending_writes
                )
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save a checkpoint to PostgreSQL.
        
        Args:
            config: Configuration containing thread_id
            checkpoint: The checkpoint data to save
            metadata: Metadata for the checkpoint
            new_versions: Version information
            
        Returns:
            Updated configuration with checkpoint_id
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        
        if not thread_id:
            raise ValueError("thread_id required in config")
        
        # Generate checkpoint_id if not provided
        checkpoint_id = checkpoint.get("id") or str(uuid.uuid4())
        parent_checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Insert or update checkpoint
                cur.execute("""
                    INSERT INTO checkpoints 
                    (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, 
                     checkpoint, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id) 
                    DO UPDATE SET 
                        checkpoint = EXCLUDED.checkpoint,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    Json(checkpoint),
                    Json(metadata)
                ))
                conn.commit()
        
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id
            }
        }
    
    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """
        List checkpoints from PostgreSQL.
        
        Args:
            config: Configuration with thread_id
            filter: Additional filters
            before: Filter for checkpoints before a certain point
            limit: Maximum number of checkpoints to return
            
        Yields:
            CheckpointTuple objects
        """
        if not config:
            return
        
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        
        if not thread_id:
            return
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = """
                    SELECT checkpoint_id, checkpoint, metadata, parent_checkpoint_id
                    FROM checkpoints
                    WHERE thread_id = %s AND checkpoint_ns = %s
                    ORDER BY created_at DESC
                """
                params = [thread_id, checkpoint_ns]
                
                if limit:
                    query += " LIMIT %s"
                    params.append(limit)
                
                cur.execute(query, params)
                
                for row in cur.fetchall():
                    checkpoint_id, checkpoint_data, metadata, parent_id = row
                    
                    parent_config = None
                    if parent_id:
                        parent_config = {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": parent_id
                            }
                        }
                    
                    yield CheckpointTuple(
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id
                            }
                        },
                        checkpoint=checkpoint_data,
                        metadata=metadata or {},
                        parent_config=parent_config
                    )
    
    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[tuple],
        task_id: str
    ) -> None:
        """
        Store pending writes.
        
        Args:
            config: Configuration with thread_id and checkpoint_id
            writes: Sequence of (channel, value) tuples
            task_id: Task identifier
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        
        if not thread_id or not checkpoint_id:
            return
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                for idx, (channel, value) in enumerate(writes):
                    cur.execute("""
                        INSERT INTO checkpoint_writes
                        (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, value)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        task_id,
                        idx,
                        channel,
                        Json(value)
                    ))
                conn.commit()