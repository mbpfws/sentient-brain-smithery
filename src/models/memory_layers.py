"""
Complete 4-Layered Memory System Models for Sentient Brain Multi-Agent System.

Implements the comprehensive memory architecture with:
1. Plans and Task-Breakdown Knowledge Base Memory Layer
2. Message Session Knowledge Base Memory Layer  
3. Documents Knowledge Base Memory Layer
4. Open-Source Git Memory Layer

Each layer supports multi-granularity chunking, vector embeddings, and rich
relational metadata for hybrid semantic search capabilities.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


# ================================
# 1. PLANS AND TASK-BREAKDOWN MEMORY LAYER
# ================================

class PlanStatus(str, Enum):
    """Status of plans and tasks."""
    DRAFT = "draft"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ON_HOLD = "on_hold"


class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKLOG = "backlog"


class GrandPlan(BaseModel):
    """Project-level goals and strategic objectives."""
    id: str = Field(default_factory=lambda: f"plan:{uuid4()}")
    title: str
    description: str
    vision: str
    success_criteria: List[str] = Field(default_factory=list)
    
    # Status and timeline
    status: PlanStatus = PlanStatus.DRAFT
    estimated_duration_days: Optional[int] = None
    start_date: Optional[datetime] = None
    target_completion: Optional[datetime] = None
    
    # Technical specifications
    tech_stack: List[str] = Field(default_factory=list)
    architecture_type: Optional[str] = None
    deployment_target: Optional[str] = None
    
    # Relationships (stored as IDs, resolved via graph queries)
    milestones: List[str] = Field(default_factory=list)
    root_tasks: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    domain_tags: List[str] = Field(default_factory=list)
    
    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str
    last_modified: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


class Task(BaseModel):
    """Hierarchical task with subtasks and dependencies."""
    id: str = Field(default_factory=lambda: f"task:{uuid4()}")
    title: str
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)
    
    # Hierarchy and relationships
    parent_plan_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = Field(default_factory=list)
    
    # Dependencies and blocking
    depends_on_task_ids: List[str] = Field(default_factory=list)
    blocked_by_task_ids: List[str] = Field(default_factory=list)
    blocks_task_ids: List[str] = Field(default_factory=list)
    
    # Status and priority
    status: PlanStatus = PlanStatus.DRAFT
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_hours: Optional[float] = None
    actual_hours: Optional[float] = None
    
    # Assignment and ownership
    assigned_agent: Optional[str] = None
    assigned_human: Optional[str] = None
    
    # Technical associations
    associated_code_files: List[str] = Field(default_factory=list)
    associated_doc_chunks: List[str] = Field(default_factory=list)
    governed_by_policies: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    domain_tags: List[str] = Field(default_factory=list)
    tech_stack_tags: List[str] = Field(default_factory=list)
    
    # Progress tracking
    completion_percentage: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str
    last_modified: datetime = Field(default_factory=datetime.utcnow)


class Milestone(BaseModel):
    """Major project milestones."""
    id: str = Field(default_factory=lambda: f"milestone:{uuid4()}")
    title: str
    description: str
    success_criteria: List[str] = Field(default_factory=list)
    
    # Associations
    plan_id: str
    required_task_ids: List[str] = Field(default_factory=list)
    deliverables: List[str] = Field(default_factory=list)
    
    # Timeline
    target_date: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    # Status
    status: PlanStatus = PlanStatus.DRAFT
    completion_percentage: float = 0.0
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    domain_tags: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str


class CompletionCriteria(BaseModel):
    """Detailed completion criteria for tasks and milestones."""
    id: str = Field(default_factory=lambda: f"criteria:{uuid4()}")
    title: str
    description: str
    criteria_type: str  # functional, non-functional, quality, etc.
    
    # Associations
    applies_to_task_id: Optional[str] = None
    applies_to_milestone_id: Optional[str] = None
    
    # Verification
    verification_method: str  # test, review, demo, etc.
    is_met: bool = False
    evidence: Optional[str] = None
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ================================
# 2. MESSAGE SESSION MEMORY LAYER
# ================================

class SessionType(str, Enum):
    """Types of message sessions."""
    USER_INTERACTION = "user_interaction"
    AGENT_COLLABORATION = "agent_collaboration"
    DEBUG_SESSION = "debug_session"
    PLANNING_SESSION = "planning_session"
    CODE_REVIEW = "code_review"


class MessageRole(str, Enum):
    """Message sender roles."""
    USER = "user"
    ORCHESTRATOR = "orchestrator"
    ARCHITECT = "architect"
    CODEBASE_AGENT = "codebase_agent"
    DEBUG_AGENT = "debug_agent"
    TASK_AGENT = "task_agent"
    DOCUMENT_AGENT = "document_agent"
    SYSTEM = "system"


class MessageSession(BaseModel):
    """Complete conversation session."""
    id: str = Field(default_factory=lambda: f"session:{uuid4()}")
    title: Optional[str] = None
    session_type: SessionType = SessionType.USER_INTERACTION
    
    # Participants
    user_id: Optional[str] = None
    participating_agents: List[str] = Field(default_factory=list)
    
    # Context and state
    project_context_id: Optional[str] = None
    current_workflow_id: Optional[str] = None
    session_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Message tracking
    message_count: int = 0
    turn_ids: List[str] = Field(default_factory=list)
    
    # Timeline
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    
    # Outcomes and results
    created_task_ids: List[str] = Field(default_factory=list)
    modified_code_files: List[str] = Field(default_factory=list)
    referenced_docs: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    session_summary: Optional[str] = None
    summary_embedding: Optional[List[float]] = None
    intent_tags: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)


class MessageTurn(BaseModel):
    """Individual message within a session."""
    id: str = Field(default_factory=lambda: f"msg:{uuid4()}")
    session_id: str
    turn_number: int
    
    # Message content
    role: MessageRole
    content: str
    content_type: str = "text"  # text, code, image, etc.
    
    # Context and intent
    intent: Optional[str] = None
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    
    # Relationships - what this message refers to or creates
    refers_to_task_ids: List[str] = Field(default_factory=list)
    refers_to_code_files: List[str] = Field(default_factory=list)
    refers_to_doc_chunks: List[str] = Field(default_factory=list)
    results_in_task_ids: List[str] = Field(default_factory=list)
    influences_policies: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    
    # Processing metadata
    processing_time_ms: Optional[float] = None
    model_used: Optional[str] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionContext(BaseModel):
    """Derived context and summary for sessions."""
    id: str = Field(default_factory=lambda: f"context:{uuid4()}")
    session_id: str
    
    # Derived insights
    main_topics: List[str] = Field(default_factory=list)
    key_decisions: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    
    # Technical context
    mentioned_technologies: List[str] = Field(default_factory=list)
    code_patterns_discussed: List[str] = Field(default_factory=list)
    
    # Vector representation
    context_summary: str
    embedding: Optional[List[float]] = None
    
    # Relationships
    related_session_ids: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ================================
# 3. DOCUMENTS MEMORY LAYER
# ================================

class DocumentType(str, Enum):
    """Types of documents."""
    OFFICIAL_DOCS = "official_docs"
    API_REFERENCE = "api_reference"
    TUTORIAL = "tutorial"
    BLOG_POST = "blog_post"
    STACKOVERFLOW = "stackoverflow"
    GITHUB_README = "github_readme"
    SPECIFICATION = "specification"
    BEST_PRACTICES = "best_practices"


class DocumentSource(BaseModel):
    """Source information for documents."""
    id: str = Field(default_factory=lambda: f"source:{uuid4()}")
    url: str
    domain: str
    source_type: DocumentType
    
    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    # Reliability and quality
    trust_score: float = 0.5  # 0.0 to 1.0
    quality_score: float = 0.5
    relevance_score: float = 0.5
    
    # Technical metadata
    tech_stack: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    
    # Scraping metadata
    scraped_at: datetime = Field(default_factory=datetime.utcnow)
    scraper_version: str = "1.0"
    content_hash: Optional[str] = None


class Document(BaseModel):
    """Complete document with metadata."""
    id: str = Field(default_factory=lambda: f"doc:{uuid4()}")
    title: str
    content: str
    
    # Source information
    source_id: str
    original_url: str
    
    # Structure and hierarchy
    document_type: DocumentType
    hierarchy_level: int = 0  # 0 = root, 1 = section, 2 = subsection, etc.
    parent_doc_id: Optional[str] = None
    
    # Content metadata
    word_count: int
    estimated_read_time_minutes: int
    language: str = "en"
    
    # Technical classification
    tech_stack: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    difficulty_level: str = "intermediate"  # beginner, intermediate, advanced
    
    # Chunking information
    chunk_ids: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    
    # Quality and applicability
    applicability_score: float = 0.5
    freshness_score: float = 0.5
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_by_agent: str


class DocChunk(BaseModel):
    """Document chunk with rich metadata."""
    id: str = Field(default_factory=lambda: f"chunk:{uuid4()}")
    document_id: str
    
    # Content
    title: Optional[str] = None
    content: str
    chunk_type: str  # section, code_example, api_method, concept, etc.
    
    # Position and structure
    chunk_index: int
    start_position: int
    end_position: int
    hierarchy_path: List[str] = Field(default_factory=list)
    
    # Relationships - what this chunk refers to or documents
    refers_to_code_files: List[str] = Field(default_factory=list)
    refers_to_tasks: List[str] = Field(default_factory=list)
    about_domains: List[str] = Field(default_factory=list)
    about_frameworks: List[str] = Field(default_factory=list)
    
    # Similarity and relationships
    similar_chunk_ids: List[str] = Field(default_factory=list)
    prerequisite_chunk_ids: List[str] = Field(default_factory=list)
    follow_up_chunk_ids: List[str] = Field(default_factory=list)
    
    # Vector and semantic information
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    extracted_entities: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Usage and quality metrics
    usage_count: int = 0
    helpfulness_score: float = 0.5
    accuracy_score: float = 0.5
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)


# ================================
# 4. OPEN-SOURCE GIT MEMORY LAYER
# ================================

class RepoType(str, Enum):
    """Types of repositories."""
    LIBRARY = "library"
    FRAMEWORK = "framework"
    APPLICATION = "application"
    TOOL = "tool"
    TEMPLATE = "template"
    EXAMPLE = "example"
    DOCUMENTATION = "documentation"


class GitRepo(BaseModel):
    """External open-source repository."""
    id: str = Field(default_factory=lambda: f"repo:{uuid4()}")
    name: str
    full_name: str  # owner/repo
    url: str
    clone_url: str
    
    # Repository metadata
    description: Optional[str] = None
    repo_type: RepoType = RepoType.LIBRARY
    primary_language: Optional[str] = None
    languages: Dict[str, int] = Field(default_factory=dict)  # language: bytes
    
    # GitHub/Git metadata
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    open_issues: int = 0
    license: Optional[str] = None
    
    # Version and activity
    default_branch: str = "main"
    latest_commit_sha: Optional[str] = None
    last_push: Optional[datetime] = None
    created_at_source: Optional[datetime] = None
    
    # Analysis metadata
    relevance_score: float = 0.5  # How relevant to our project
    quality_score: float = 0.5    # Code quality assessment
    adoption_score: float = 0.5   # Community adoption
    
    # Technical classification
    tech_stack: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    
    # Ingestion metadata
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    last_analyzed: datetime = Field(default_factory=datetime.utcnow)
    analysis_depth: str = "shallow"  # shallow, medium, deep
    
    # Files and features
    file_ids: List[str] = Field(default_factory=list)
    feature_ids: List[str] = Field(default_factory=list)


class RepoFile(BaseModel):
    """File within a repository."""
    id: str = Field(default_factory=lambda: f"file:{uuid4()}")
    repo_id: str
    
    # File information
    file_path: str
    file_name: str
    file_type: str  # extension
    file_size: int
    
    # Content
    content: Optional[str] = None
    content_hash: Optional[str] = None
    
    # Git metadata
    last_modified: Optional[datetime] = None
    commit_sha: Optional[str] = None
    
    # Analysis
    language: Optional[str] = None
    complexity_score: Optional[float] = None
    lines_of_code: Optional[int] = None
    
    # Chunking
    chunk_ids: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RepoChunk(BaseModel):
    """Code chunk from repository file."""
    id: str = Field(default_factory=lambda: f"repochunk:{uuid4()}")
    repo_id: str
    file_id: str
    
    # Content and position
    content: str
    chunk_type: str  # function, class, method, block, comment, etc.
    start_line: int
    end_line: int
    
    # Code structure
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    method_name: Optional[str] = None
    
    # Relationships - similarity and improvement potential
    similar_to_local_chunks: List[str] = Field(default_factory=list)
    improves_local_files: List[str] = Field(default_factory=list)
    improves_local_tasks: List[str] = Field(default_factory=list)
    
    # Learning and inspiration
    demonstrates_patterns: List[str] = Field(default_factory=list)
    teaches_concepts: List[str] = Field(default_factory=list)
    
    # Vector and semantic information
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    
    # Quality and applicability
    code_quality_score: float = 0.5
    applicability_score: float = 0.5
    complexity_score: float = 0.5
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Feature(BaseModel):
    """High-level capability or feature extracted from repositories."""
    id: str = Field(default_factory=lambda: f"feature:{uuid4()}")
    name: str
    description: str
    
    # Source information
    source_repo_ids: List[str] = Field(default_factory=list)
    source_chunk_ids: List[str] = Field(default_factory=list)
    
    # Feature classification
    feature_type: str  # authentication, api, ui_component, algorithm, etc.
    complexity_level: str = "medium"  # simple, medium, complex
    
    # Implementation details
    required_dependencies: List[str] = Field(default_factory=list)
    implementation_patterns: List[str] = Field(default_factory=list)
    
    # Relationships - how this feature relates to our project
    inspires_plans: List[str] = Field(default_factory=list)
    inspires_tasks: List[str] = Field(default_factory=list)
    applicable_to_domains: List[str] = Field(default_factory=list)
    
    # Vector and metadata
    embedding: Optional[List[float]] = None
    semantic_tags: List[str] = Field(default_factory=list)
    
    # Usage and validation
    adoption_examples: List[str] = Field(default_factory=list)
    validation_score: float = 0.5
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    discovered_by_agent: str


# ================================
# CROSS-LAYER RELATIONSHIP MODELS
# ================================

class RelationshipType(str, Enum):
    """Types of relationships between memory layer entities."""
    # Hierarchical relationships
    HAS_SUBTASK = "HAS_SUBTASK"
    HAS_CHUNK = "HAS_CHUNK"
    HAS_TURN = "HAS_TURN"
    HAS_FILE = "HAS_FILE"
    
    # Dependency relationships
    DEPENDS_ON = "DEPENDS_ON"
    BLOCKED_BY = "BLOCKED_BY"
    REQUIRES = "REQUIRES"
    
    # Reference relationships
    REFERS_TO = "REFERS_TO"
    DOCUMENTS = "DOCUMENTS"
    EXPLAINS = "EXPLAINS"
    ABOUT = "ABOUT"
    
    # Outcome relationships
    RESULTS_IN = "RESULTS_IN"
    CREATES = "CREATES"
    MODIFIES = "MODIFIES"
    
    # Governance relationships
    GOVERNED_BY = "GOVERNED_BY"
    APPLIES_TO = "APPLIES_TO"
    ENFORCES = "ENFORCES"
    
    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    IMPROVES = "IMPROVES"
    INSPIRES = "INSPIRES"
    INFLUENCES = "INFLUENCES"
    
    # Workflow relationships
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    SUPPORTS = "SUPPORTS"
    ENABLES = "ENABLES"


class MemoryRelationship(BaseModel):
    """Rich relationship between any two memory layer entities."""
    id: str = Field(default_factory=lambda: f"rel:{uuid4()}")
    
    # Relationship definition
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    
    # Relationship metadata
    strength: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    bidirectional: bool = False
    
    # Context and reasoning
    context: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    evidence: List[str] = Field(default_factory=list)
    
    # Lifecycle
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by_agent: str
    last_validated: Optional[datetime] = None
    validation_score: float = 1.0
    
    # Domain and tags
    domain_tags: List[str] = Field(default_factory=list)
    semantic_tags: List[str] = Field(default_factory=list)


# ================================
# UNIFIED MEMORY QUERY MODELS
# ================================

class MemoryQuery(BaseModel):
    """Unified query across all memory layers."""
    query_text: str
    query_embedding: Optional[List[float]] = None
    
    # Layer targeting
    target_layers: List[str] = Field(default_factory=list)  # empty = all layers
    
    # Filtering
    domain_filter: List[str] = Field(default_factory=list)
    tech_stack_filter: List[str] = Field(default_factory=list)
    entity_type_filter: List[str] = Field(default_factory=list)
    
    # Search parameters
    similarity_threshold: float = 0.7
    max_results_per_layer: int = 10
    include_relationships: bool = True
    relationship_depth: int = 2
    
    # Context
    current_project_id: Optional[str] = None
    current_session_id: Optional[str] = None
    user_context: Dict[str, Any] = Field(default_factory=dict)


class MemoryQueryResult(BaseModel):
    """Results from unified memory query."""
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    
    # Results by layer
    plan_results: List[Dict[str, Any]] = Field(default_factory=list)
    task_results: List[Dict[str, Any]] = Field(default_factory=list)
    message_results: List[Dict[str, Any]] = Field(default_factory=list)
    document_results: List[Dict[str, Any]] = Field(default_factory=list)
    code_results: List[Dict[str, Any]] = Field(default_factory=list)
    repo_results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Relationships and connections
    cross_layer_relationships: List[MemoryRelationship] = Field(default_factory=list)
    
    # Query metadata
    total_results: int
    processing_time_ms: float
    layers_searched: List[str]
    
    # Ranking and relevance
    relevance_scores: Dict[str, float] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow) 