"""
Knowledge-related data models for the Sentient Brain Multi-Agent System.

Defines structures for knowledge representation, code analysis, documentation,
and memory layers using SurrealDB's multi-model capabilities.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class KnowledgeNodeType(str, Enum):
    """Types of knowledge nodes in the system."""
    CODE_CHUNK = "code_chunk"
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    DOCUMENTATION = "documentation"
    TASK = "task"
    PROJECT = "project"
    DEPENDENCY = "dependency"
    API_ENDPOINT = "api_endpoint"
    DATABASE_SCHEMA = "database_schema"


class EmbeddingModel(str, Enum):
    """Supported embedding models."""
    GEMINI = "gemini"
    LOCAL_HF = "local_hf"
    OPENAI = "openai"


class KnowledgeNode(BaseModel):
    """Base knowledge node structure for SurrealDB storage."""
    id: str = Field(default_factory=lambda: f"knowledge:{uuid4()}")
    node_type: KnowledgeNodeType
    title: str
    content: str
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Vector embeddings for semantic search
    embedding: Optional[List[float]] = None
    embedding_model: Optional[EmbeddingModel] = None
    
    # Hierarchy and relationships
    parent_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    
    # Quality and relevance metrics
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    usage_count: int = Field(default=0, ge=0)
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)


class CodeChunk(BaseModel):
    """Specialized knowledge node for code analysis."""
    id: str = Field(default_factory=lambda: f"code:{uuid4()}")
    file_path: str
    start_line: int
    end_line: int
    language: str
    content: str
    
    # AST analysis results
    ast_data: Optional[Dict[str, Any]] = None
    functions: List[str] = Field(default_factory=list)
    classes: List[str] = Field(default_factory=list)
    imports: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Semantic analysis
    complexity_score: Optional[float] = None
    maintainability_index: Optional[float] = None
    technical_debt_score: Optional[float] = None
    
    # Relationships
    calls_functions: List[str] = Field(default_factory=list)
    called_by: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)
    
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentChunk(BaseModel):
    """Knowledge node for documentation and external content."""
    id: str = Field(default_factory=lambda: f"doc:{uuid4()}")
    source_url: Optional[str] = None
    title: str
    content: str
    chunk_number: int = 0
    
    # Document metadata
    doc_type: str = "markdown"  # markdown, html, pdf, etc.
    author: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Processing results
    summary: Optional[str] = None
    key_concepts: List[str] = Field(default_factory=list)
    
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TaskItem(BaseModel):
    """Task representation for project planning and execution."""
    id: str = Field(default_factory=lambda: f"task:{uuid4()}")
    title: str
    description: str
    
    # Task hierarchy
    parent_task_id: Optional[str] = None
    subtasks: List[str] = Field(default_factory=list)
    
    # Status and priority
    status: str = "todo"  # todo, in_progress, done, blocked
    priority: str = "medium"  # low, medium, high, critical
    
    # Dependencies and relationships
    depends_on: List[str] = Field(default_factory=list)
    blocks: List[str] = Field(default_factory=list)
    related_code: List[str] = Field(default_factory=list)
    related_docs: List[str] = Field(default_factory=list)
    
    # Assignee and timeline
    assigned_agent: Optional[str] = None
    estimated_effort: Optional[int] = None  # in minutes
    actual_effort: Optional[int] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None


class RelationType(str, Enum):
    """Types of relationships between knowledge nodes."""
    DEPENDS_ON = "depends_on"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    REFERENCES = "references"
    DOCUMENTS = "documents"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    CONFLICTS_WITH = "conflicts_with"
    REPLACES = "replaces"


class KnowledgeRelation(BaseModel):
    """Relationship between knowledge nodes."""
    id: str = Field(default_factory=lambda: f"relation:{uuid4()}")
    source_id: str
    target_id: str
    relation_type: RelationType
    
    # Relationship metadata
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Context and provenance
    context: Optional[str] = None
    discovered_by: Optional[str] = None  # Which agent discovered this relation
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectContext(BaseModel):
    """Project-level context and metadata."""
    id: str = Field(default_factory=lambda: f"project:{uuid4()}")
    name: str
    description: str
    
    # Project structure
    root_path: str
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    # Project stats
    total_files: int = 0
    total_lines: int = 0
    code_quality_score: Optional[float] = None
    
    # Domain configuration
    domains: List[str] = Field(default_factory=list)
    tech_stack: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class QueryResult(BaseModel):
    """Result structure for knowledge queries."""
    nodes: List[KnowledgeNode] = Field(default_factory=list)
    relations: List[KnowledgeRelation] = Field(default_factory=list)
    total_count: int = 0
    query_time_ms: float = 0.0
    
    # Context and metadata
    query: str
    semantic_similarity_threshold: Optional[float] = None
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    
    # Ranking and relevance
    max_relevance_score: Optional[float] = None
    min_relevance_score: Optional[float] = None 