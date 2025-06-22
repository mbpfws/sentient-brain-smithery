"""
Agent-related data models for the Sentient Brain Multi-Agent System.

Defines the structure and behavior of different agent types, their configurations,
messages, and responses within the system.
"""
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4


class AgentType(str, Enum):
    """Types of agents in the multi-agent system."""
    ULTRA_ORCHESTRATOR = "ultra_orchestrator"
    ARCHITECT = "architect"
    CODEBASE_KNOWLEDGE_MEMORY = "codebase_knowledge_memory"
    DEBUG_REFACTOR = "debug_refactor"
    PLAN_TASKS_MEMORY = "plan_tasks_memory"
    DOCUMENTS_MEMORY = "documents_memory"
    CLIENT_SIDE_AI = "client_side_ai"


class AgentState(str, Enum):
    """Current state of an agent."""
    IDLE = "idle"
    BUSY = "busy"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"


class MessageType(str, Enum):
    """Types of messages between agents."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    INFORMATION_QUERY = "information_query"
    INFORMATION_RESPONSE = "information_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    COORDINATION = "coordination"


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    agent_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_type: AgentType
    name: str
    description: str
    version: str = "1.0.0"
    
    # Capabilities and settings
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300
    retry_limit: int = 3
    
    # LLM configuration
    llm_model: str = "llama-3.1-70b-versatile"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    # Specialization
    specialized_domains: List[str] = Field(default_factory=list)
    required_tools: List[str] = Field(default_factory=list)
    memory_size: int = 1000  # Number of items to keep in memory
    
    # Integration settings
    database_config: Dict[str, Any] = Field(default_factory=dict)
    external_apis: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None


class AgentMessage(BaseModel):
    """Message passed between agents."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    message_type: MessageType
    sender_id: str
    receiver_id: str
    
    # Message content
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Threading and context
    thread_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Timing and priority
    priority: int = 3  # 1 = highest, 5 = lowest
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Delivery tracking
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None


class AgentResponse(BaseModel):
    """Response from an agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    success: bool = True
    
    # Response content
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Error handling
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Processing information
    processing_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    confidence_score: Optional[float] = None
    
    # Context and tracing
    request_id: Optional[str] = None
    conversation_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class AgentCapability(BaseModel):
    """Capability definition for an agent."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
    # Performance characteristics
    average_response_time_ms: float = 1000.0
    success_rate: float = 0.95
    concurrent_limit: int = 1
    
    # Requirements
    required_permissions: List[str] = Field(default_factory=list)
    required_resources: Dict[str, Any] = Field(default_factory=dict)
    
    # Dependencies
    depends_on_agents: List[str] = Field(default_factory=list)
    depends_on_tools: List[str] = Field(default_factory=list)


class AgentMemory(BaseModel):
    """Memory structure for agent state persistence."""
    agent_id: str
    memory_type: str  # short_term, long_term, working, episodic
    
    # Memory content
    key: str
    value: Any
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    
    # Expiration
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Relationships
    related_memories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AgentPerformanceMetrics(BaseModel):
    """Performance tracking for agents."""
    agent_id: str
    metric_type: str
    
    # Core metrics
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time_ms: float = 0.0
    
    # Resource usage
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    tokens_consumed: int = 0
    api_calls_made: int = 0
    
    # Quality metrics
    user_satisfaction_score: float = 0.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Time period
    period_start: datetime
    period_end: datetime
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)


class AgentHealthStatus(BaseModel):
    """Health status monitoring for agents."""
    agent_id: str
    status: AgentState
    
    # Health indicators
    is_healthy: bool = True
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    uptime_seconds: float = 0.0
    
    # Resource status
    cpu_available: bool = True
    memory_available: bool = True
    network_available: bool = True
    database_available: bool = True
    
    # Current load
    active_tasks: int = 0
    queued_tasks: int = 0
    load_percentage: float = 0.0
    
    # Error tracking
    recent_errors: List[str] = Field(default_factory=list)
    error_count_last_hour: int = 0
    
    # Diagnostics
    version: str = "1.0.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    diagnostic_info: Dict[str, Any] = Field(default_factory=dict) 