"""
SurrealDB Service for unified data layer management.

Provides comprehensive interface to SurrealDB's multi-model capabilities
supporting document, graph, and vector storage for the multi-agent system.
"""
import asyncio
from typing import Dict, List, Any, Optional, Union
import logging
from surrealdb import Surreal
import os

logger = logging.getLogger(__name__)


class SurrealDBService:
    """
    Unified SurrealDB service providing multi-model data capabilities.
    
    Supports document storage, graph relationships, and vector embeddings
    in a single unified data layer for the multi-agent system.
    """
    
    def __init__(
        self,
        url: str = "ws://localhost:8000/rpc",
        username: str = "root",
        password: str = "root",
        namespace: str = "sentient_brain",
        database: str = "multi_agent"
    ):
        self.url = url
        self.username = username
        self.password = password
        self.namespace = namespace
        self.database = database
        self.db = Surreal()
        self._connected = False
        
    async def connect(self) -> bool:
        """
        Establish connection to SurrealDB instance.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            await self.db.connect(self.url)
            await self.db.signin({"user": self.username, "pass": self.password})
            await self.db.use(self.namespace, self.database)
            self._connected = True
            logger.info(f"Connected to SurrealDB: {self.namespace}/{self.database}")
            
            # Initialize schema if needed
            await self._initialize_schema()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to SurrealDB: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from SurrealDB."""
        if self._connected:
            await self.db.close()
            self._connected = False
            logger.info("Disconnected from SurrealDB")
    
    async def _initialize_schema(self):
        """Initialize database schema for multi-agent system."""
        try:
            # Create tables and define relationships
            schema_queries = [
                # Knowledge nodes table
                """
                DEFINE TABLE knowledge SCHEMAFULL;
                DEFINE FIELD node_type ON knowledge TYPE string;
                DEFINE FIELD title ON knowledge TYPE string;
                DEFINE FIELD content ON knowledge TYPE string;
                DEFINE FIELD embedding ON knowledge TYPE array<float>;
                DEFINE FIELD metadata ON knowledge TYPE object;
                DEFINE FIELD created_at ON knowledge TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_node_type ON knowledge COLUMNS node_type;
                DEFINE INDEX idx_embedding ON knowledge COLUMNS embedding MTREE DIMENSION 1536;
                """,
                
                # Code chunks table
                """
                DEFINE TABLE code SCHEMAFULL;
                DEFINE FIELD file_path ON code TYPE string;
                DEFINE FIELD language ON code TYPE string;
                DEFINE FIELD content ON code TYPE string;
                DEFINE FIELD ast_data ON code TYPE object;
                DEFINE FIELD functions ON code TYPE array<string>;
                DEFINE FIELD classes ON code TYPE array<string>;
                DEFINE FIELD embedding ON code TYPE array<float>;
                DEFINE FIELD created_at ON code TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_file_path ON code COLUMNS file_path;
                DEFINE INDEX idx_language ON code COLUMNS language;
                """,
                
                # Workflows table
                """
                DEFINE TABLE workflows SCHEMAFULL;
                DEFINE FIELD workflow_type ON workflows TYPE string;
                DEFINE FIELD status ON workflows TYPE string;
                DEFINE FIELD initial_input ON workflows TYPE object;
                DEFINE FIELD current_state ON workflows TYPE object;
                DEFINE FIELD involved_agents ON workflows TYPE array<string>;
                DEFINE FIELD created_at ON workflows TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_status ON workflows COLUMNS status;
                DEFINE INDEX idx_workflow_type ON workflows COLUMNS workflow_type;
                """,
                
                # Projects table
                """
                DEFINE TABLE projects SCHEMAFULL;
                DEFINE FIELD name ON projects TYPE string;
                DEFINE FIELD description ON projects TYPE string;
                DEFINE FIELD root_path ON projects TYPE string;
                DEFINE FIELD languages ON projects TYPE array<string>;
                DEFINE FIELD frameworks ON projects TYPE array<string>;
                DEFINE FIELD created_at ON projects TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_name ON projects COLUMNS name;
                """,
                
                # Relations table for graph relationships
                """
                DEFINE TABLE relations SCHEMAFULL;
                DEFINE FIELD source_id ON relations TYPE string;
                DEFINE FIELD target_id ON relations TYPE string;
                DEFINE FIELD relation_type ON relations TYPE string;
                DEFINE FIELD weight ON relations TYPE float DEFAULT 1.0;
                DEFINE FIELD confidence ON relations TYPE float DEFAULT 0.5;
                DEFINE FIELD created_at ON relations TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_relation_type ON relations COLUMNS relation_type;
                """,
            ]
            
            for query in schema_queries:
                await self.db.query(query)
            
            logger.info("Schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            raise
    
    async def create_record(self, table: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new record in the specified table.
        
        Args:
            table: Table name
            data: Record data
            
        Returns:
            Created record ID or None if failed
        """
        try:
            result = await self.db.create(table, data)
            if result and len(result) > 0:
                record_id = result[0].get("id", "")
                logger.debug(f"Created record in {table}: {record_id}")
                return record_id
            return None
            
        except Exception as e:
            logger.error(f"Failed to create record in {table}: {e}")
            return None
    
    async def get_record(self, table: str, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a record by ID.
        
        Args:
            table: Table name
            record_id: Record ID
            
        Returns:
            Record data or None if not found
        """
        try:
            result = await self.db.select(f"{table}:{record_id}")
            if result and len(result) > 0:
                return result[0]
            return None
            
        except Exception as e:
            logger.error(f"Failed to get record {record_id} from {table}: {e}")
            return None
    
    async def update_record(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
        """
        Update an existing record.
        
        Args:
            table: Table name
            record_id: Record ID
            data: Updated data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.db.update(f"{table}:{record_id}", data)
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to update record {record_id} in {table}: {e}")
            return False
    
    async def delete_record(self, table: str, record_id: str) -> bool:
        """
        Delete a record by ID.
        
        Args:
            table: Table name
            record_id: Record ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = await self.db.delete(f"{table}:{record_id}")
            return result is not None
            
        except Exception as e:
            logger.error(f"Failed to delete record {record_id} from {table}: {e}")
            return False
    
    async def query_records(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom SurrealQL query.
        
        Args:
            query: SurrealQL query string
            params: Query parameters
            
        Returns:
            Query results
        """
        try:
            if params:
                result = await self.db.query(query, params)
            else:
                result = await self.db.query(query)
            
            if result and len(result) > 0:
                return result[0].get("result", [])
            return []
            
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
    async def vector_search(
        self,
        table: str,
        embedding: List[float],
        limit: int = 10,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Args:
            table: Table to search
            embedding: Query embedding vector
            limit: Maximum results to return
            threshold: Similarity threshold
            
        Returns:
            Similar records with similarity scores
        """
        try:
            # Convert embedding to SurrealDB format
            embedding_str = str(embedding)
            
            query = f"""
                SELECT *,
                    vector::similarity::cosine(embedding, {embedding_str}) AS similarity
                FROM {table}
                WHERE embedding IS NOT NONE
                ORDER BY similarity DESC
                LIMIT {limit}
            """
            
            results = await self.query_records(query)
            
            # Filter by threshold
            filtered_results = [
                r for r in results 
                if r.get("similarity", 0) >= threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to perform vector search: {e}")
            return []
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a relationship between two records.
        
        Args:
            source_id: Source record ID
            target_id: Target record ID
            relation_type: Type of relationship
            metadata: Additional relationship metadata
            
        Returns:
            Relationship ID or None if failed
        """
        try:
            relation_data = {
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "metadata": metadata or {}
            }
            
            return await self.create_record("relations", relation_data)
            
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return None
    
    async def get_related_records(
        self,
        record_id: str,
        relation_type: Optional[str] = None,
        direction: str = "out"  # "out", "in", or "both"
    ) -> List[Dict[str, Any]]:
        """
        Get records related to a specific record.
        
        Args:
            record_id: Source record ID
            relation_type: Specific relation type to filter
            direction: Direction of relationships to follow
            
        Returns:
            Related records
        """
        try:
            base_query = "SELECT * FROM relations WHERE "
            
            if direction == "out":
                condition = f"source_id = '{record_id}'"
            elif direction == "in":
                condition = f"target_id = '{record_id}'"
            else:  # both
                condition = f"source_id = '{record_id}' OR target_id = '{record_id}'"
            
            if relation_type:
                condition += f" AND relation_type = '{relation_type}'"
            
            query = base_query + condition
            relations = await self.query_records(query)
            
            # Get the actual records
            related_records = []
            for relation in relations:
                if direction == "out" or (direction == "both" and relation["source_id"] == record_id):
                    target_record = await self.get_record("knowledge", relation["target_id"])
                    if target_record:
                        target_record["relationship"] = relation
                        related_records.append(target_record)
                
                if direction == "in" or (direction == "both" and relation["target_id"] == record_id):
                    source_record = await self.get_record("knowledge", relation["source_id"])
                    if source_record:
                        source_record["relationship"] = relation
                        related_records.append(source_record)
            
            return related_records
            
        except Exception as e:
            logger.error(f"Failed to get related records: {e}")
            return []
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status and database information."""
        return {
            "connected": self._connected,
            "url": self.url,
            "namespace": self.namespace,
            "database": self.database,
            "version": await self._get_version() if self._connected else None
        }
    
    async def _get_version(self) -> Optional[str]:
        """Get SurrealDB version."""
        try:
            result = await self.db.version()
            return result if result else "unknown"
        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            return None 