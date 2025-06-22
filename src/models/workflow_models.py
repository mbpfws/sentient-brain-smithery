"""
Workflow-related data models for the Sentient Brain Multi-Agent System.

Defines structures for orchestrating complex multi-agent workflows,
project phases, and user intent processing using LangGraph patterns.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class ProjectPhase(str, Enum):
    """Project development phases."""
    DISCOVERY = "discovery"
    REQUIREMENTS = "requirements"
    ARCHITECTURE = "architecture"
    DEVELOPMENT = "development"
    TESTING = "testing"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"


class UserIntentType(str, Enum):
    """Types of user intents that can be processed."""
    NEW_PROJECT = "new_project"
    EXISTING_PROJECT = "existing_project"
    DEBUG_REQUEST = "debug_request"
    REFACTOR_REQUEST = "refactor_request"
    DOCUMENTATION_REQUEST = "documentation_request"
    ANALYSIS_REQUEST = "analysis_request"
    GENERAL_QUERY = "general_query"


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentWorkflowType(str, Enum):
    """Types of agent workflows in the system."""
    ORCHESTRATION = "orchestration"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_ANALYSIS = "code_analysis"
    DEBUG_REFACTOR = "debug_refactor"
    DOCUMENTATION = "documentation"
    TASK_PLANNING = "task_planning"
    MEMORY_RETRIEVAL = "memory_retrieval"


class UserIntent(BaseModel):
    """User intent analysis and processing."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    raw_query: str
    intent_type: UserIntentType
    confidence: float = Field(ge=0.0, le=1.0)
    entities: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    user_technical_level: str = "intermediate"  # novice, intermediate, expert
    preferred_mode: str = "guided"  # guided, autonomous
    required_clarifications: List[str] = Field(default_factory=list)
    processed_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowNode(BaseModel):
    """Individual node in a workflow."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: str  # agent, decision, action, memory
    agent_type: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """State and progress of a workflow execution."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_type: AgentWorkflowType
    status: WorkflowStatus = WorkflowStatus.PENDING
    initial_input: Dict[str, Any] = Field(default_factory=dict)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    final_output: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Workflow execution tracking
    nodes: List[WorkflowNode] = Field(default_factory=list)
    current_node: Optional[str] = None
    completed_nodes: List[str] = Field(default_factory=list)
    failed_nodes: List[str] = Field(default_factory=list)
    
    # Agent involvement
    involved_agents: List[str] = Field(default_factory=list)
    agent_states: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Timing and metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Memory and persistence
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    persistent_memory: Dict[str, Any] = Field(default_factory=dict)


class AgentTask(BaseModel):
    """Individual task within a workflow assigned to an agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_id: str
    agent_type: str
    task_type: str
    priority: int = 1  # 1 = highest, 5 = lowest
    
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[Dict[str, Any]] = None
    
    status: str = "pending"  # pending, running, completed, failed
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    dependencies: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    
    error_message: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowTemplate(BaseModel):
    """Template for creating standardized workflows."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    workflow_type: AgentWorkflowType
    version: str = "1.0.0"
    
    # Template structure
    node_templates: List[WorkflowNode] = Field(default_factory=list)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    required_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    
    # Configuration
    timeout_minutes: int = 30
    max_retries: int = 3
    parallel_execution: bool = False
    
    # Metadata
    created_by: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    usage_count: int = 0


class LangGraphState(BaseModel):
    """State object for LangGraph integration."""
    workflow_id: str
    current_step: str
    step_count: int = 0
    max_steps: int = 100
    
    # Data flow
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    intermediate_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Agent coordination
    active_agents: List[str] = Field(default_factory=list)
    agent_communications: List[Dict[str, Any]] = Field(default_factory=list)
    shared_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Control flow
    next_steps: List[str] = Field(default_factory=list)
    conditional_branches: Dict[str, Any] = Field(default_factory=dict)
    loop_state: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Timing
    step_start_time: Optional[datetime] = None
    total_execution_time: float = 0.0


class AgentCollaboration(BaseModel):
    """Model for agent-to-agent collaboration."""
    id: str = Field(default_factory=lambda: f"collab:{uuid4()}")
    requesting_agent: str
    target_agent: str
    collaboration_type: str  # consultation, delegation, peer_review, etc.
    
    # Request details
    request_context: Dict[str, Any] = Field(default_factory=dict)
    expected_deliverable: str
    priority: str = "medium"  # low, medium, high, urgent
    
    # Response
    response_data: Optional[Dict[str, Any]] = None
    status: str = "requested"  # requested, accepted, in_progress, completed, rejected
    
    # Timing
    requested_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class MemoryLayer(BaseModel):
    """Memory layer abstraction for agent persistence."""
    id: str = Field(default_factory=lambda: f"memory:{uuid4()}")
    layer_type: str  # codebase, documentation, tasks, project
    agent_id: str
    
    # Memory content
    short_term_memory: Dict[str, Any] = Field(default_factory=dict)
    long_term_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Context and retrieval
    active_context: List[str] = Field(default_factory=list)
    recent_queries: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance metrics
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    relevance_threshold: float = 0.3
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectMetrics(BaseModel):
    """Metrics for project development progress."""
    id: str = Field(default_factory=lambda: f"metrics:{uuid4()}")
    project_id: str
    
    # Development metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    blocked_tasks: int = 0
    
    # Code metrics
    lines_of_code: int = 0
    functions_count: int = 0
    classes_count: int = 0
    complexity_score: Optional[float] = None
    
    # Quality metrics
    test_coverage: Optional[float] = None
    code_quality_score: Optional[float] = None
    technical_debt_score: Optional[float] = None
    
    # Agent activity
    active_agents: List[str] = Field(default_factory=list)
    agent_utilization: Dict[str, float] = Field(default_factory=dict)
    
    # Timeline
    estimated_completion: Optional[datetime] = None
    actual_completion: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DecisionPoint(BaseModel):
    """Decision points in workflows requiring human or agent input."""
    id: str = Field(default_factory=lambda: f"decision:{uuid4()}")
    workflow_id: str
    node_id: str
    
    # Decision details
    decision_type: str  # approval, choice, configuration, etc.
    question: str
    options: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision maker
    assigned_to: str  # agent_id or "human"
    priority: str = "medium"
    
    # Resolution
    decision_made: Optional[Dict[str, Any]] = None
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None


class ExecutionResult(BaseModel):
    """Result of workflow or agent execution."""
    id: str = Field(default_factory=lambda: f"result:{uuid4()}")
    workflow_id: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Result data
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)  # File paths or IDs
    
    # Performance metrics
    execution_time_ms: float = 0.0
    tokens_used: Optional[int] = None
    api_calls_made: int = 0
    
    # Error information
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Context
    input_context: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow) 