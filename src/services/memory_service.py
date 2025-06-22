"""
Comprehensive Memory Service for 4-Layered Knowledge Graph.

Manages all memory layers with vector embeddings, graph relationships,
and hybrid semantic search capabilities across:
1. Plans and Task-Breakdown Memory Layer
2. Message Session Memory Layer  
3. Documents Memory Layer
4. Open-Source Git Memory Layer
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import hashlib
import json
from sentence_transformers import SentenceTransformer

from ..models.memory_layers import (
    # Plans layer
    GrandPlan, Task, Milestone, CompletionCriteria,
    # Messages layer  
    MessageSession, MessageTurn, SessionContext,
    # Documents layer
    Document, DocChunk, DocumentSource,
    # Git layer
    GitRepo, RepoFile, RepoChunk, Feature,
    # Cross-layer
    MemoryRelationship, RelationshipType, MemoryQuery, MemoryQueryResult
)
from .surreal_service import SurrealDBService
from .groq_service import GroqLLMService

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Unified memory service managing all 4 knowledge layers with 
    vector embeddings and rich graph relationships.
    """
    
    def __init__(
        self,
        db_service: SurrealDBService,
        llm_service: GroqLLMService,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.db = db_service
        self.llm = llm_service
        
        # Initialize embedding model
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dimension = 384  # for all-MiniLM-L6-v2
        
        # Layer table mappings
        self.layer_tables = {
            "plans": ["grand_plans", "tasks", "milestones", "completion_criteria"],
            "messages": ["message_sessions", "message_turns", "session_contexts"], 
            "documents": ["documents", "doc_chunks", "document_sources"],
            "git": ["git_repos", "repo_files", "repo_chunks", "features"],
            "relationships": ["memory_relationships"]
        }
    
    async def initialize_schema(self):
        """Initialize comprehensive schema for all memory layers."""
        try:
            schema_queries = [
                # ================================
                # PLANS AND TASKS LAYER SCHEMA
                # ================================
                """
                DEFINE TABLE grand_plans SCHEMAFULL;
                DEFINE FIELD title ON grand_plans TYPE string;
                DEFINE FIELD description ON grand_plans TYPE string;
                DEFINE FIELD vision ON grand_plans TYPE string;
                DEFINE FIELD status ON grand_plans TYPE string;
                DEFINE FIELD tech_stack ON grand_plans TYPE array<string>;
                DEFINE FIELD embedding ON grand_plans TYPE array<float>;
                DEFINE FIELD domain_tags ON grand_plans TYPE array<string>;
                DEFINE FIELD created_by_agent ON grand_plans TYPE string;
                DEFINE FIELD created_at ON grand_plans TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_plans_status ON grand_plans COLUMNS status;
                DEFINE INDEX idx_plans_domain ON grand_plans COLUMNS domain_tags;
                DEFINE INDEX idx_plans_embedding ON grand_plans COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                """
                DEFINE TABLE tasks SCHEMAFULL;
                DEFINE FIELD title ON tasks TYPE string;
                DEFINE FIELD description ON tasks TYPE string;
                DEFINE FIELD parent_plan_id ON tasks TYPE option<string>;
                DEFINE FIELD parent_task_id ON tasks TYPE option<string>;
                DEFINE FIELD status ON tasks TYPE string;
                DEFINE FIELD priority ON tasks TYPE string;
                DEFINE FIELD assigned_agent ON tasks TYPE option<string>;
                DEFINE FIELD completion_percentage ON tasks TYPE float DEFAULT 0.0;
                DEFINE FIELD embedding ON tasks TYPE array<float>;
                DEFINE FIELD domain_tags ON tasks TYPE array<string>;
                DEFINE FIELD tech_stack_tags ON tasks TYPE array<string>;
                DEFINE FIELD created_by_agent ON tasks TYPE string;
                DEFINE FIELD created_at ON tasks TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_tasks_status ON tasks COLUMNS status;
                DEFINE INDEX idx_tasks_priority ON tasks COLUMNS priority;
                DEFINE INDEX idx_tasks_domain ON tasks COLUMNS domain_tags;
                DEFINE INDEX idx_tasks_embedding ON tasks COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                """
                DEFINE TABLE milestones SCHEMAFULL;
                DEFINE FIELD title ON milestones TYPE string;
                DEFINE FIELD description ON milestones TYPE string;
                DEFINE FIELD plan_id ON milestones TYPE string;
                DEFINE FIELD status ON milestones TYPE string;
                DEFINE FIELD target_date ON milestones TYPE option<datetime>;
                DEFINE FIELD completion_percentage ON milestones TYPE float DEFAULT 0.0;
                DEFINE FIELD embedding ON milestones TYPE array<float>;
                DEFINE FIELD domain_tags ON milestones TYPE array<string>;
                DEFINE FIELD created_by_agent ON milestones TYPE string;
                DEFINE FIELD created_at ON milestones TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_milestones_status ON milestones COLUMNS status;
                DEFINE INDEX idx_milestones_embedding ON milestones COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                # ================================
                # MESSAGE SESSION LAYER SCHEMA
                # ================================
                """
                DEFINE TABLE message_sessions SCHEMAFULL;
                DEFINE FIELD title ON message_sessions TYPE option<string>;
                DEFINE FIELD session_type ON message_sessions TYPE string;
                DEFINE FIELD user_id ON message_sessions TYPE option<string>;
                DEFINE FIELD participating_agents ON message_sessions TYPE array<string>;
                DEFINE FIELD project_context_id ON message_sessions TYPE option<string>;
                DEFINE FIELD message_count ON message_sessions TYPE int DEFAULT 0;
                DEFINE FIELD session_summary ON message_sessions TYPE option<string>;
                DEFINE FIELD summary_embedding ON message_sessions TYPE array<float>;
                DEFINE FIELD intent_tags ON message_sessions TYPE array<string>;
                DEFINE FIELD domain_tags ON message_sessions TYPE array<string>;
                DEFINE FIELD started_at ON message_sessions TYPE datetime DEFAULT time::now();
                DEFINE FIELD last_activity ON message_sessions TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_sessions_type ON message_sessions COLUMNS session_type;
                DEFINE INDEX idx_sessions_user ON message_sessions COLUMNS user_id;
                DEFINE INDEX idx_sessions_embedding ON message_sessions COLUMNS summary_embedding MTREE DIMENSION 384;
                """,
                
                """
                DEFINE TABLE message_turns SCHEMAFULL;
                DEFINE FIELD session_id ON message_turns TYPE string;
                DEFINE FIELD turn_number ON message_turns TYPE int;
                DEFINE FIELD role ON message_turns TYPE string;
                DEFINE FIELD content ON message_turns TYPE string;
                DEFINE FIELD intent ON message_turns TYPE option<string>;
                DEFINE FIELD confidence_score ON message_turns TYPE option<float>;
                DEFINE FIELD refers_to_task_ids ON message_turns TYPE array<string>;
                DEFINE FIELD refers_to_code_files ON message_turns TYPE array<string>;
                DEFINE FIELD embedding ON message_turns TYPE array<float>;
                DEFINE FIELD semantic_tags ON message_turns TYPE array<string>;
                DEFINE FIELD timestamp ON message_turns TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_turns_session ON message_turns COLUMNS session_id;
                DEFINE INDEX idx_turns_role ON message_turns COLUMNS role;
                DEFINE INDEX idx_turns_embedding ON message_turns COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                # ================================
                # DOCUMENTS LAYER SCHEMA
                # ================================
                """
                DEFINE TABLE document_sources SCHEMAFULL;
                DEFINE FIELD url ON document_sources TYPE string;
                DEFINE FIELD domain ON document_sources TYPE string;
                DEFINE FIELD source_type ON document_sources TYPE string;
                DEFINE FIELD trust_score ON document_sources TYPE float DEFAULT 0.5;
                DEFINE FIELD quality_score ON document_sources TYPE float DEFAULT 0.5;
                DEFINE FIELD tech_stack ON document_sources TYPE array<string>;
                DEFINE FIELD scraped_at ON document_sources TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_sources_domain ON document_sources COLUMNS domain;
                DEFINE INDEX idx_sources_type ON document_sources COLUMNS source_type;
                DEFINE INDEX idx_sources_tech ON document_sources COLUMNS tech_stack;
                """,
                
                """
                DEFINE TABLE documents SCHEMAFULL;
                DEFINE FIELD title ON documents TYPE string;
                DEFINE FIELD content ON documents TYPE string;
                DEFINE FIELD source_id ON documents TYPE string;
                DEFINE FIELD original_url ON documents TYPE string;
                DEFINE FIELD document_type ON documents TYPE string;
                DEFINE FIELD hierarchy_level ON documents TYPE int DEFAULT 0;
                DEFINE FIELD word_count ON documents TYPE int;
                DEFINE FIELD tech_stack ON documents TYPE array<string>;
                DEFINE FIELD difficulty_level ON documents TYPE string DEFAULT 'intermediate';
                DEFINE FIELD embedding ON documents TYPE array<float>;
                DEFINE FIELD semantic_tags ON documents TYPE array<string>;
                DEFINE FIELD applicability_score ON documents TYPE float DEFAULT 0.5;
                DEFINE FIELD processed_by_agent ON documents TYPE string;
                DEFINE FIELD created_at ON documents TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_docs_type ON documents COLUMNS document_type;
                DEFINE INDEX idx_docs_tech ON documents COLUMNS tech_stack;
                DEFINE INDEX idx_docs_embedding ON documents COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                """
                DEFINE TABLE doc_chunks SCHEMAFULL;
                DEFINE FIELD document_id ON doc_chunks TYPE string;
                DEFINE FIELD title ON doc_chunks TYPE option<string>;
                DEFINE FIELD content ON doc_chunks TYPE string;
                DEFINE FIELD chunk_type ON doc_chunks TYPE string;
                DEFINE FIELD chunk_index ON doc_chunks TYPE int;
                DEFINE FIELD refers_to_code_files ON doc_chunks TYPE array<string>;
                DEFINE FIELD about_domains ON doc_chunks TYPE array<string>;
                DEFINE FIELD about_frameworks ON doc_chunks TYPE array<string>;
                DEFINE FIELD embedding ON doc_chunks TYPE array<float>;
                DEFINE FIELD semantic_tags ON doc_chunks TYPE array<string>;
                DEFINE FIELD usage_count ON doc_chunks TYPE int DEFAULT 0;
                DEFINE FIELD helpfulness_score ON doc_chunks TYPE float DEFAULT 0.5;
                DEFINE FIELD created_at ON doc_chunks TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_chunks_doc ON doc_chunks COLUMNS document_id;
                DEFINE INDEX idx_chunks_type ON doc_chunks COLUMNS chunk_type;
                DEFINE INDEX idx_chunks_domains ON doc_chunks COLUMNS about_domains;
                DEFINE INDEX idx_chunks_embedding ON doc_chunks COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                # ================================
                # GIT REPOSITORIES LAYER SCHEMA
                # ================================
                """
                DEFINE TABLE git_repos SCHEMAFULL;
                DEFINE FIELD name ON git_repos TYPE string;
                DEFINE FIELD full_name ON git_repos TYPE string;
                DEFINE FIELD url ON git_repos TYPE string;
                DEFINE FIELD description ON git_repos TYPE option<string>;
                DEFINE FIELD repo_type ON git_repos TYPE string;
                DEFINE FIELD primary_language ON git_repos TYPE option<string>;
                DEFINE FIELD stars ON git_repos TYPE int DEFAULT 0;
                DEFINE FIELD forks ON git_repos TYPE int DEFAULT 0;
                DEFINE FIELD relevance_score ON git_repos TYPE float DEFAULT 0.5;
                DEFINE FIELD quality_score ON git_repos TYPE float DEFAULT 0.5;
                DEFINE FIELD tech_stack ON git_repos TYPE array<string>;
                DEFINE FIELD frameworks ON git_repos TYPE array<string>;
                DEFINE FIELD use_cases ON git_repos TYPE array<string>;
                DEFINE FIELD ingested_at ON git_repos TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_repos_language ON git_repos COLUMNS primary_language;
                DEFINE INDEX idx_repos_tech ON git_repos COLUMNS tech_stack;
                DEFINE INDEX idx_repos_relevance ON git_repos COLUMNS relevance_score;
                """,
                
                """
                DEFINE TABLE repo_chunks SCHEMAFULL;
                DEFINE FIELD repo_id ON repo_chunks TYPE string;
                DEFINE FIELD file_id ON repo_chunks TYPE string;
                DEFINE FIELD content ON repo_chunks TYPE string;
                DEFINE FIELD chunk_type ON repo_chunks TYPE string;
                DEFINE FIELD start_line ON repo_chunks TYPE int;
                DEFINE FIELD end_line ON repo_chunks TYPE int;
                DEFINE FIELD function_name ON repo_chunks TYPE option<string>;
                DEFINE FIELD class_name ON repo_chunks TYPE option<string>;
                DEFINE FIELD similar_to_local_chunks ON repo_chunks TYPE array<string>;
                DEFINE FIELD improves_local_files ON repo_chunks TYPE array<string>;
                DEFINE FIELD demonstrates_patterns ON repo_chunks TYPE array<string>;
                DEFINE FIELD embedding ON repo_chunks TYPE array<float>;
                DEFINE FIELD semantic_tags ON repo_chunks TYPE array<string>;
                DEFINE FIELD code_quality_score ON repo_chunks TYPE float DEFAULT 0.5;
                DEFINE FIELD applicability_score ON repo_chunks TYPE float DEFAULT 0.5;
                DEFINE FIELD created_at ON repo_chunks TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_repochunks_repo ON repo_chunks COLUMNS repo_id;
                DEFINE INDEX idx_repochunks_type ON repo_chunks COLUMNS chunk_type;
                DEFINE INDEX idx_repochunks_embedding ON repo_chunks COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                """
                DEFINE TABLE features SCHEMAFULL;
                DEFINE FIELD name ON features TYPE string;
                DEFINE FIELD description ON features TYPE string;
                DEFINE FIELD feature_type ON features TYPE string;
                DEFINE FIELD complexity_level ON features TYPE string DEFAULT 'medium';
                DEFINE FIELD source_repo_ids ON features TYPE array<string>;
                DEFINE FIELD required_dependencies ON features TYPE array<string>;
                DEFINE FIELD implementation_patterns ON features TYPE array<string>;
                DEFINE FIELD inspires_plans ON features TYPE array<string>;
                DEFINE FIELD applicable_to_domains ON features TYPE array<string>;
                DEFINE FIELD embedding ON features TYPE array<float>;
                DEFINE FIELD semantic_tags ON features TYPE array<string>;
                DEFINE FIELD validation_score ON features TYPE float DEFAULT 0.5;
                DEFINE FIELD discovered_by_agent ON features TYPE string;
                DEFINE FIELD created_at ON features TYPE datetime DEFAULT time::now();
                DEFINE INDEX idx_features_type ON features COLUMNS feature_type;
                DEFINE INDEX idx_features_domains ON features COLUMNS applicable_to_domains;
                DEFINE INDEX idx_features_embedding ON features COLUMNS embedding MTREE DIMENSION 384;
                """,
                
                # ================================
                # CROSS-LAYER RELATIONSHIPS SCHEMA
                # ================================
                """
                DEFINE TABLE memory_relationships SCHEMAFULL;
                DEFINE FIELD source_id ON memory_relationships TYPE string;
                DEFINE FIELD target_id ON memory_relationships TYPE string;
                DEFINE FIELD relationship_type ON memory_relationships TYPE string;
                DEFINE FIELD strength ON memory_relationships TYPE float DEFAULT 1.0;
                DEFINE FIELD confidence ON memory_relationships TYPE float DEFAULT 1.0;
                DEFINE FIELD bidirectional ON memory_relationships TYPE bool DEFAULT false;
                DEFINE FIELD reasoning ON memory_relationships TYPE option<string>;
                DEFINE FIELD created_by_agent ON memory_relationships TYPE string;
                DEFINE FIELD created_at ON memory_relationships TYPE datetime DEFAULT time::now();
                DEFINE FIELD validation_score ON memory_relationships TYPE float DEFAULT 1.0;
                DEFINE FIELD domain_tags ON memory_relationships TYPE array<string>;
                DEFINE INDEX idx_rels_source ON memory_relationships COLUMNS source_id;
                DEFINE INDEX idx_rels_target ON memory_relationships COLUMNS target_id;
                DEFINE INDEX idx_rels_type ON memory_relationships COLUMNS relationship_type;
                DEFINE INDEX idx_rels_strength ON memory_relationships COLUMNS strength;
                """,
            ]
            
            for query in schema_queries:
                await self.db.query_records(query)
            
            logger.info("Memory service schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory schema: {e}")
            raise
    
    # ================================
    # EMBEDDING AND VECTORIZATION
    # ================================
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text."""
        try:
            embedding = self.embedder.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dimension
    
    def generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    # ================================
    # PLANS AND TASKS LAYER OPERATIONS
    # ================================
    
    async def create_grand_plan(self, plan: GrandPlan) -> Optional[str]:
        """Create a new grand plan with embedding."""
        try:
            # Generate embedding for plan content
            plan_text = f"{plan.title} {plan.description} {plan.vision}"
            plan.embedding = self.generate_embedding(plan_text)
            
            # Store in database
            plan_id = await self.db.create_record("grand_plans", plan.dict())
            
            logger.info(f"Created grand plan: {plan_id}")
            return plan_id
            
        except Exception as e:
            logger.error(f"Failed to create grand plan: {e}")
            return None
    
    async def create_task(
        self, 
        task: Task, 
        auto_link: bool = True
    ) -> Optional[str]:
        """Create a new task with embedding and optional auto-linking."""
        try:
            # Generate embedding
            task_text = f"{task.title} {task.description}"
            task.embedding = self.generate_embedding(task_text)
            
            # Store task
            task_id = await self.db.create_record("tasks", task.dict())
            
            # Auto-link to related entities if requested
            if auto_link and task_id:
                await self._auto_link_task(task_id, task)
            
            logger.info(f"Created task: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    async def _auto_link_task(self, task_id: str, task: Task):
        """Automatically create relationships for a task."""
        try:
            # Link to parent plan if specified
            if task.parent_plan_id:
                await self.create_relationship(
                    task.parent_plan_id,
                    task_id,
                    RelationshipType.HAS_SUBTASK,
                    created_by_agent=task.created_by_agent
                )
            
            # Link to parent task if specified
            if task.parent_task_id:
                await self.create_relationship(
                    task.parent_task_id,
                    task_id,
                    RelationshipType.HAS_SUBTASK,
                    created_by_agent=task.created_by_agent
                )
            
            # Create dependency relationships
            for dep_task_id in task.depends_on_task_ids:
                await self.create_relationship(
                    task_id,
                    dep_task_id,
                    RelationshipType.DEPENDS_ON,
                    created_by_agent=task.created_by_agent
                )
            
        except Exception as e:
            logger.error(f"Failed to auto-link task {task_id}: {e}")
    
    # ================================
    # MESSAGE SESSION LAYER OPERATIONS
    # ================================
    
    async def create_message_session(self, session: MessageSession) -> Optional[str]:
        """Create a new message session."""
        try:
            session_id = await self.db.create_record("message_sessions", session.dict())
            logger.info(f"Created message session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Failed to create message session: {e}")
            return None
    
    async def add_message_turn(
        self, 
        turn: MessageTurn,
        auto_link: bool = True
    ) -> Optional[str]:
        """Add a message turn with embedding and auto-linking."""
        try:
            # Generate embedding
            turn.embedding = self.generate_embedding(turn.content)
            
            # Store turn
            turn_id = await self.db.create_record("message_turns", turn.dict())
            
            # Update session message count
            if turn_id:
                await self._update_session_count(turn.session_id)
                
                # Auto-link to referenced entities
                if auto_link:
                    await self._auto_link_message_turn(turn_id, turn)
            
            logger.info(f"Added message turn: {turn_id}")
            return turn_id
            
        except Exception as e:
            logger.error(f"Failed to add message turn: {e}")
            return None
    
    async def _update_session_count(self, session_id: str):
        """Update message count for a session."""
        try:
            query = f"""
                UPDATE {session_id} SET 
                    message_count = (SELECT count() FROM message_turns WHERE session_id = '{session_id}'),
                    last_activity = time::now()
            """
            await self.db.query_records(query)
        except Exception as e:
            logger.error(f"Failed to update session count: {e}")
    
    async def _auto_link_message_turn(self, turn_id: str, turn: MessageTurn):
        """Auto-link message turn to referenced entities."""
        try:
            # Link to session
            await self.create_relationship(
                turn.session_id,
                turn_id,
                RelationshipType.HAS_TURN,
                created_by_agent="message_agent"
            )
            
            # Link to referenced tasks
            for task_id in turn.refers_to_task_ids:
                await self.create_relationship(
                    turn_id,
                    task_id,
                    RelationshipType.REFERS_TO,
                    created_by_agent="message_agent"
                )
            
        except Exception as e:
            logger.error(f"Failed to auto-link message turn {turn_id}: {e}")
    
    # ================================
    # DOCUMENTS LAYER OPERATIONS
    # ================================
    
    async def ingest_document(
        self, 
        document: Document,
        auto_chunk: bool = True
    ) -> Optional[str]:
        """Ingest a document with embedding and optional chunking."""
        try:
            # Generate embedding for full document
            document.embedding = self.generate_embedding(document.content)
            
            # Store document
            doc_id = await self.db.create_record("documents", document.dict())
            
            # Auto-chunk if requested
            if auto_chunk and doc_id:
                await self._auto_chunk_document(doc_id, document)
            
            logger.info(f"Ingested document: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            return None
    
    async def _auto_chunk_document(self, doc_id: str, document: Document):
        """Automatically chunk a document into smaller pieces."""
        try:
            # Simple paragraph-based chunking (can be enhanced)
            paragraphs = document.content.split('\n\n')
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) < 50:  # Skip very short paragraphs
                    continue
                
                chunk = DocChunk(
                    document_id=doc_id,
                    content=paragraph.strip(),
                    chunk_type="paragraph",
                    chunk_index=i,
                    start_position=0,  # Would calculate actual position
                    end_position=len(paragraph),
                    about_domains=document.tech_stack,
                    semantic_tags=document.semantic_tags
                )
                
                # Generate embedding for chunk
                chunk.embedding = self.generate_embedding(chunk.content)
                
                # Store chunk
                chunk_id = await self.db.create_record("doc_chunks", chunk.dict())
                
                # Link chunk to document
                if chunk_id:
                    await self.create_relationship(
                        doc_id,
                        chunk_id,
                        RelationshipType.HAS_CHUNK,
                        created_by_agent=document.processed_by_agent
                    )
            
        except Exception as e:
            logger.error(f"Failed to auto-chunk document {doc_id}: {e}")
    
    # ================================
    # GIT REPOSITORIES LAYER OPERATIONS
    # ================================
    
    async def ingest_git_repo(self, repo: GitRepo) -> Optional[str]:
        """Ingest a Git repository with metadata."""
        try:
            repo_id = await self.db.create_record("git_repos", repo.dict())
            logger.info(f"Ingested git repo: {repo_id}")
            return repo_id
        except Exception as e:
            logger.error(f"Failed to ingest git repo: {e}")
            return None
    
    async def add_repo_chunk(
        self, 
        chunk: RepoChunk,
        auto_link: bool = True
    ) -> Optional[str]:
        """Add a repository code chunk with embedding."""
        try:
            # Generate embedding
            chunk.embedding = self.generate_embedding(chunk.content)
            
            # Store chunk
            chunk_id = await self.db.create_record("repo_chunks", chunk.dict())
            
            # Auto-link to similar local code if requested
            if auto_link and chunk_id:
                await self._find_similar_local_code(chunk_id, chunk)
            
            logger.info(f"Added repo chunk: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Failed to add repo chunk: {e}")
            return None
    
    async def _find_similar_local_code(self, chunk_id: str, chunk: RepoChunk):
        """Find and link similar local code chunks."""
        try:
            # This would use vector similarity search to find local code
            # that's similar to this repository chunk
            similar_results = await self.vector_search(
                "code_chunks",  # Assuming we have local code chunks
                chunk.embedding,
                limit=5,
                threshold=0.8
            )
            
            for result in similar_results:
                await self.create_relationship(
                    chunk_id,
                    result["id"],
                    RelationshipType.SIMILAR_TO,
                    strength=result["similarity"],
                    created_by_agent="git_agent"
                )
                
        except Exception as e:
            logger.error(f"Failed to find similar local code for {chunk_id}: {e}")
    
    # ================================
    # CROSS-LAYER RELATIONSHIP OPERATIONS
    # ================================
    
    async def create_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        confidence: float = 1.0,
        reasoning: Optional[str] = None,
        created_by_agent: str = "system"
    ) -> Optional[str]:
        """Create a relationship between any two entities."""
        try:
            relationship = MemoryRelationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                reasoning=reasoning,
                created_by_agent=created_by_agent
            )
            
            rel_id = await self.db.create_record("memory_relationships", relationship.dict())
            logger.debug(f"Created relationship: {source_id} -> {target_id} ({relationship_type})")
            return rel_id
            
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return None
    
    async def get_relationships(
        self,
        entity_id: str,
        relationship_types: Optional[List[RelationshipType]] = None,
        direction: str = "both"  # "out", "in", "both"
    ) -> List[MemoryRelationship]:
        """Get relationships for an entity."""
        try:
            conditions = []
            
            if direction == "out":
                conditions.append(f"source_id = '{entity_id}'")
            elif direction == "in":
                conditions.append(f"target_id = '{entity_id}'")
            else:  # both
                conditions.append(f"(source_id = '{entity_id}' OR target_id = '{entity_id}')")
            
            if relationship_types:
                type_conditions = [f"relationship_type = '{rt.value}'" for rt in relationship_types]
                conditions.append(f"({' OR '.join(type_conditions)})")
            
            query = f"SELECT * FROM memory_relationships WHERE {' AND '.join(conditions)}"
            results = await self.db.query_records(query)
            
            return [MemoryRelationship(**result) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to get relationships for {entity_id}: {e}")
            return []
    
    # ================================
    # UNIFIED HYBRID SEARCH
    # ================================
    
    async def hybrid_search(self, query: MemoryQuery) -> MemoryQueryResult:
        """Perform hybrid search across all memory layers."""
        start_time = datetime.utcnow()
        
        try:
            # Generate query embedding if not provided
            if not query.query_embedding:
                query.query_embedding = self.generate_embedding(query.query_text)
            
            result = MemoryQueryResult()
            
            # Search each layer if not filtered
            layers_to_search = query.target_layers or ["plans", "messages", "documents", "git"]
            
            if "plans" in layers_to_search:
                result.plan_results = await self._search_plans_layer(query)
                result.task_results = await self._search_tasks_layer(query)
            
            if "messages" in layers_to_search:
                result.message_results = await self._search_messages_layer(query)
            
            if "documents" in layers_to_search:
                result.document_results = await self._search_documents_layer(query)
            
            if "git" in layers_to_search:
                result.repo_results = await self._search_git_layer(query)
            
            # Find cross-layer relationships
            result.cross_layer_relationships = await self._find_cross_layer_relationships(result)
            
            # Calculate totals and metadata
            result.total_results = (
                len(result.plan_results) + len(result.task_results) +
                len(result.message_results) + len(result.document_results) +
                len(result.repo_results)
            )
            result.layers_searched = layers_to_search
            
            end_time = datetime.utcnow()
            result.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            logger.info(f"Hybrid search completed: {result.total_results} results in {result.processing_time_ms:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {e}")
            return MemoryQueryResult()
    
    async def _search_plans_layer(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Search plans and tasks layer."""
        try:
            # Vector similarity search on plans
            results = await self.db.vector_search(
                "grand_plans",
                query.query_embedding,
                limit=query.max_results_per_layer,
                threshold=query.similarity_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search plans layer: {e}")
            return []
    
    async def _search_tasks_layer(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Search tasks specifically."""
        try:
            results = await self.db.vector_search(
                "tasks",
                query.query_embedding,
                limit=query.max_results_per_layer,
                threshold=query.similarity_threshold
            )
            return results
        except Exception as e:
            logger.error(f"Failed to search tasks layer: {e}")
            return []
    
    async def _search_messages_layer(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Search message sessions and turns."""
        try:
            # Search both sessions and individual turns
            session_results = await self.db.vector_search(
                "message_sessions",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            turn_results = await self.db.vector_search(
                "message_turns",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            return session_results + turn_results
        except Exception as e:
            logger.error(f"Failed to search messages layer: {e}")
            return []
    
    async def _search_documents_layer(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Search documents and chunks."""
        try:
            # Search both full documents and chunks
            doc_results = await self.db.vector_search(
                "documents",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            chunk_results = await self.db.vector_search(
                "doc_chunks",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            return doc_results + chunk_results
        except Exception as e:
            logger.error(f"Failed to search documents layer: {e}")
            return []
    
    async def _search_git_layer(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Search Git repositories and code chunks."""
        try:
            # Search repo chunks and features
            chunk_results = await self.db.vector_search(
                "repo_chunks",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            feature_results = await self.db.vector_search(
                "features",
                query.query_embedding,
                limit=query.max_results_per_layer // 2,
                threshold=query.similarity_threshold
            )
            
            return chunk_results + feature_results
        except Exception as e:
            logger.error(f"Failed to search git layer: {e}")
            return []
    
    async def _find_cross_layer_relationships(self, result: MemoryQueryResult) -> List[MemoryRelationship]:
        """Find relationships between entities in search results."""
        try:
            # Collect all entity IDs from results
            entity_ids = []
            
            for layer_results in [result.plan_results, result.task_results, 
                                 result.message_results, result.document_results, 
                                 result.repo_results]:
                entity_ids.extend([r.get("id") for r in layer_results if r.get("id")])
            
            if not entity_ids:
                return []
            
            # Find relationships between these entities
            id_list = "', '".join(entity_ids)
            query = f"""
                SELECT * FROM memory_relationships 
                WHERE source_id IN ['{id_list}'] OR target_id IN ['{id_list}']
                ORDER BY strength DESC
                LIMIT 50
            """
            
            rel_results = await self.db.query_records(query)
            return [MemoryRelationship(**rel) for rel in rel_results]
            
        except Exception as e:
            logger.error(f"Failed to find cross-layer relationships: {e}")
            return []
    
    # ================================
    # ANALYTICS AND INSIGHTS
    # ================================
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the memory system."""
        try:
            stats = {}
            
            # Count entities in each layer
            for layer, tables in self.layer_tables.items():
                layer_stats = {}
                for table in tables:
                    count_query = f"SELECT count() as count FROM {table}"
                    count_result = await self.db.query_records(count_query)
                    layer_stats[table] = count_result[0]["count"] if count_result else 0
                stats[layer] = layer_stats
            
            # Relationship statistics
            rel_query = """
                SELECT relationship_type, count() as count 
                FROM memory_relationships 
                GROUP BY relationship_type
            """
            rel_results = await self.db.query_records(rel_query)
            stats["relationships_by_type"] = {r["relationship_type"]: r["count"] for r in rel_results}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {}
    
    async def analyze_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """Analyze potential knowledge gaps in the memory system."""
        try:
            gaps = []
            
            # Find tasks without associated documentation
            query = """
                SELECT t.id, t.title, t.domain_tags
                FROM tasks t
                WHERE t.associated_doc_chunks = []
                AND t.status != 'completed'
            """
            undocumented_tasks = await self.db.query_records(query)
            
            if undocumented_tasks:
                gaps.append({
                    "type": "undocumented_tasks",
                    "count": len(undocumented_tasks),
                    "items": undocumented_tasks
                })
            
            # Find isolated entities (no relationships)
            # This would require more complex queries...
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze knowledge gaps: {e}")
            return [] 