"""
Main FastAPI application for the Sentient Brain Multi-Agent System.

This application serves as the central API gateway for the multi-agent system,
integrating SurrealDB as the unified data layer and coordinating between
specialized agents for intelligent code development assistance.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any, Optional, List
import os
from datetime import datetime

# Import our models and services
from .models.agent_models import AgentConfig, AgentResponse, AgentType
from .models.workflow_models import UserIntent, WorkflowState, UserIntentType
from .models.knowledge_models import KnowledgeNode, ProjectContext
from .services.surreal_service import SurrealDBService
from .services.groq_service import GroqLLMService
from .agents.orchestrator import UltraOrchestratorAgent
from .workflows.agent_workflow import AgentWorkflowEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global services
db_service: Optional[SurrealDBService] = None
llm_service: Optional[GroqLLMService] = None
orchestrator: Optional[UltraOrchestratorAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global db_service, llm_service, orchestrator
    
    logger.info("Starting Sentient Brain Multi-Agent System...")
    
    try:
        # Initialize services
        db_service = SurrealDBService(
            url=os.getenv("SURREAL_URL", "ws://localhost:8000/rpc"),
            username=os.getenv("SURREAL_USER", "root"),
            password=os.getenv("SURREAL_PASS", "root"),
            namespace=os.getenv("SURREAL_NAMESPACE", "sentient_brain"),
            database=os.getenv("SURREAL_DATABASE", "multi_agent")
        )
        
        llm_service = GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        )
        
        # Connect to database
        connected = await db_service.connect()
        if not connected:
            raise Exception("Failed to connect to SurrealDB")
        
        # Initialize orchestrator
        orchestrator_config = AgentConfig(
            agent_type=AgentType.ULTRA_ORCHESTRATOR,
            name="Ultra Orchestrator",
            description="Central intelligence hub for multi-agent coordination",
            llm_model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        )
        
        orchestrator = UltraOrchestratorAgent(
            config=orchestrator_config,
            db_service=db_service,
            llm_service=llm_service
        )
        
        logger.info("Sentient Brain Multi-Agent System started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("Shutting down Sentient Brain Multi-Agent System...")
        if db_service:
            await db_service.disconnect()


# Create FastAPI app with lifecycle management
app = FastAPI(
    title="Sentient Brain Multi-Agent System",
    description="""
    Advanced AI Code Developer system leveraging multi-agent architecture 
    with SurrealDB unified data layer. Provides intelligent code development 
    assistance from concept to deployment.
    
    ## Features
    * 🧠 **Ultra Orchestrator** - Central intelligence and workflow coordination
    * 🏗️ **Architect Agent** - System design and architecture planning  
    * 📚 **Knowledge Memory** - Code analysis and codebase understanding
    * 🔧 **Debug/Refactor** - Code improvement and troubleshooting
    * 📝 **Document Memory** - Documentation generation and management
    * 🗄️ **SurrealDB** - Unified multi-model data layer
    * ⚡ **Groq API** - High-performance LLM inference
    
    ## Architecture
    Built on proven multi-agent patterns with LangGraph orchestration,
    ensuring scalable, maintainable, and intelligent code development workflows.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> SurrealDBService:
    """Dependency to get database service."""
    if not db_service:
        raise HTTPException(status_code=500, detail="Database service not initialized")
    return db_service


def get_orchestrator() -> UltraOrchestratorAgent:
    """Dependency to get orchestrator agent."""
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
    return orchestrator


@app.get("/")
async def root():
    """Root endpoint with system information."""
    status = await db_service.get_connection_status() if db_service else {"connected": False}
    
    return {
        "message": "Sentient Brain Multi-Agent System",
        "version": "1.0.0",
        "status": "operational",
        "database": status,
        "agents": {
            "orchestrator": "active" if orchestrator else "inactive",
            "available_types": [agent_type.value for agent_type in AgentType]
        },
        "capabilities": [
            "intent_disambiguation",
            "project_context_analysis", 
            "workflow_orchestration",
            "code_analysis",
            "architecture_design",
            "debug_assistance",
            "documentation_generation"
        ]
    }


@app.post("/api/v1/query", response_model=AgentResponse)
async def process_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    orchestrator_agent: UltraOrchestratorAgent = Depends(get_orchestrator)
):
    """
    Process user query through the multi-agent system.
    
    This is the main entry point for user interactions. The Ultra Orchestrator
    will analyze intent, coordinate agents, and return comprehensive results.
    """
    try:
        response = await orchestrator_agent.process_user_query(query, context)
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/projects", response_model=Dict[str, Any])
async def create_project(
    name: str,
    description: str = "",
    root_path: str = ".",
    languages: List[str] = None,
    frameworks: List[str] = None,
    db: SurrealDBService = Depends(get_db)
):
    """Create a new project context."""
    try:
        project_context = ProjectContext(
            name=name,
            description=description,
            root_path=root_path,
            languages=languages or [],
            frameworks=frameworks or []
        )
        
        project_id = await db.create_record("projects", project_context.dict())
        
        return {
            "project_id": project_id,
            "status": "created",
            "project": project_context.dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/projects/{project_id}")
async def get_project(
    project_id: str,
    db: SurrealDBService = Depends(get_db)
):
    """Get project information by ID."""
    try:
        project_data = await db.get_record("projects", project_id)
        if not project_data:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {"project": project_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workflows")
async def list_workflows(
    status: Optional[str] = None,
    workflow_type: Optional[str] = None,
    limit: int = 50,
    db: SurrealDBService = Depends(get_db)
):
    """List workflows with optional filtering."""
    try:
        query = "SELECT * FROM workflows"
        conditions = []
        
        if status:
            conditions.append(f"status = '{status}'")
        if workflow_type:
            conditions.append(f"workflow_type = '{workflow_type}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY created_at DESC LIMIT {limit}"
        
        workflows = await db.query_records(query)
        
        return {
            "workflows": workflows,
            "count": len(workflows)
        }
        
    except Exception as e:
        logger.error(f"Error listing workflows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workflows/{workflow_id}")
async def get_workflow(
    workflow_id: str,
    db: SurrealDBService = Depends(get_db)
):
    """Get workflow details by ID."""
    try:
        workflow_data = await db.get_record("workflows", workflow_id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {"workflow": workflow_data}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/knowledge")
async def create_knowledge_node(
    title: str,
    content: str,
    node_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    db: SurrealDBService = Depends(get_db)
):
    """Create a new knowledge node."""
    try:
        knowledge_node = KnowledgeNode(
            title=title,
            content=content,
            node_type=node_type,
            metadata=metadata or {}
        )
        
        node_id = await db.create_record("knowledge", knowledge_node.dict())
        
        return {
            "node_id": node_id,
            "status": "created",
            "node": knowledge_node.dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating knowledge node: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/knowledge/search")
async def search_knowledge(
    query: str,
    node_type: Optional[str] = None,
    limit: int = 20,
    db: SurrealDBService = Depends(get_db)
):
    """Search knowledge nodes by content."""
    try:
        # Simple text search - in production would use vector search
        search_query = f"SELECT * FROM knowledge WHERE content CONTAINS '{query}'"
        
        if node_type:
            search_query += f" AND node_type = '{node_type}'"
        
        search_query += f" LIMIT {limit}"
        
        results = await db.query_records(search_query)
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        db_status = await db_service.get_connection_status() if db_service else {"connected": False}
        
        health_status = {
            "status": "healthy" if db_status.get("connected") else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": db_status,
                "orchestrator": orchestrator is not None,
                "llm_service": llm_service is not None
            }
        }
        
        if not db_status.get("connected"):
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/system/initialize")
async def initialize_system(
    background_tasks: BackgroundTasks,
    db: SurrealDBService = Depends(get_db)
):
    """Initialize the system with default data and configurations."""
    try:
        # This would typically be run once during deployment
        # Create default project contexts, knowledge templates, etc.
        
        def initialize_background():
            logger.info("Running system initialization in background...")
            # Add initialization logic here
        
        background_tasks.add_task(initialize_background)
        
        return {
            "status": "initialization_started",
            "message": "System initialization running in background"
        }
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
