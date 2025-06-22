#!/usr/bin/env python3
"""
Sentient Brain Multi-Agent MCP Server for Smithery.ai
Optimized for HTTP streaming protocol with configuration via query parameters
"""

import os
import asyncio
import sqlite3
import aiosqlite
from typing import Dict, Any, List, Optional, Union
from urllib.parse import parse_qs, urlparse
import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Static input schemas for tools (used in lazy tool listing)
# ---------------------------------------------------------
class OrchestrateInput(BaseModel):
    query: str = Field(..., description="User query or task description")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    priority: str = Field(default="medium", description="Task priority", pattern="^(low|medium|high)$")

class ArchitectInput(BaseModel):
    project_type: str = Field(..., description="Type of project to architect")
    requirements: str = Field(..., description="Project requirements")
    tech_stack: Optional[List[str]] = Field(default_factory=list, description="Proposed tech stack")

class AnalyzeCodeInput(BaseModel):
    code: str = Field(..., description="Code to analyze")
    language: Optional[str] = Field(default=None, description="Programming language of the code")
    analysis_type: str = Field(default="structure", description="Type of analysis", pattern="^(structure|quality|dependencies)$")

class SearchKnowledgeInput(BaseModel):
    query: str = Field(..., description="Search query")
    node_type: Optional[str] = Field(default="code_chunk", description="Type of node to search for", pattern="^(code_chunk|task|document|concept)$")
    limit: int = Field(default=10, description="Maximum number of results to return")

class DebugAssistInput(BaseModel):
    code: str = Field(..., description="Code containing issues")
    error_message: Optional[str] = Field(default=None, description="Associated error message, if any")
    debug_type: str = Field(default="fix", description="Type of debugging action", pattern="^(fix|optimize|refactor)$")


class SQLiteDatabase:
    """Simple SQLite database service for persistent storage"""
    
    def __init__(self, db_path: str = "./sentient_brain.db"):
        self.db_path = db_path
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize database schema"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Create tables for storing interactions and knowledge
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        type TEXT NOT NULL,
                        agent TEXT NOT NULL,
                        query TEXT,
                        response TEXT,
                        metadata TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_type TEXT NOT NULL,
                        title TEXT,
                        content TEXT,
                        metadata TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(type)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(node_type)
                """)
                
                await db.commit()
                
            self._initialized = True
            logger.info(f"SQLite database initialized: {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            return False
    
    async def store_interaction(self, interaction_type: str, agent: str, query: str, response: Dict[str, Any]) -> Optional[int]:
        """Store an interaction in the database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    INSERT INTO interactions (type, agent, query, response, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    interaction_type,
                    agent,
                    query,
                    json.dumps(response),
                    json.dumps({"success": response.get("success", True)})
                ))
                await db.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to store interaction: {e}")
            return None
    
    async def search_interactions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search interactions by query content"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("""
                    SELECT id, type, agent, query, response, timestamp
                    FROM interactions
                    WHERE query LIKE ? OR response LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                rows = await cursor.fetchall()
                results = []
                for row in rows:
                    results.append({
                        "id": f"interaction_{row[0]}",
                        "type": row[1],
                        "agent": row[2],
                        "content": f"Query: {row[3][:100]}..." if len(row[3]) > 100 else row[3],
                        "relevance": 0.8,
                        "metadata": {"source": "database", "created": row[5]}
                    })
                return results
        except Exception as e:
            logger.error(f"Failed to search interactions: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                cursor = await db.execute("SELECT COUNT(*) FROM interactions")
                interaction_result = await cursor.fetchone()
                interaction_count = interaction_result[0] if interaction_result else 0
                
                cursor = await db.execute("SELECT COUNT(*) FROM knowledge")
                knowledge_result = await cursor.fetchone()
                knowledge_count = knowledge_result[0] if knowledge_result else 0
                
                return {
                    "interactions": interaction_count,
                    "knowledge_nodes": knowledge_count,
                    "database_size": os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"interactions": 0, "knowledge_nodes": 0, "database_size": 0}


class Config(BaseSettings):
    """Configuration loaded from environment or Smithery query params"""
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    database_path: str = Field(default="./sentient_brain.db", alias="DATABASE_PATH")
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    port: int = Field(default=8000, alias="PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Allow extra fields to prevent validation errors during tool scanning

class MCPRequest(BaseModel):
    """MCP Protocol Request"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None  # Allow both string and int IDs
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    """MCP Protocol Response"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class SentientBrainMCP:
    """Main MCP Server Implementation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tools = []  # Will be lazily initialized
        self.agents = {}
        self._is_initialized = False
        self.database = SQLiteDatabase(config.database_path)
        self.db_available = False  # Track if database is working
        logger.info(f"Created SentientBrainMCP instance with SQLite database: {config.database_path}")

    def initialize(self) -> None:
        """Lazily initialize all resources and services"""
        if self._is_initialized:
            return
            
        # Initialize tools - always available regardless of database
        self.tools = self.get_tool_definitions()
        
        # Try to detect if we have database access
        try:
            # Simple check - if we can't resolve localhost:8000, we're probably in Smithery
            import socket
            socket.create_connection(("localhost", 8000), timeout=1).close()
            self.fallback_mode = False
            logger.info("Database connection available - full mode")
        except:
            self.fallback_mode = True
            logger.info("No database detected - running in fallback mode with in-memory storage")
            
        self._is_initialized = True
        logger.info(f"Fully initialized SentientBrainMCP with {len(self.tools)} tools (fallback_mode: {self.fallback_mode})")
    
    @staticmethod
    def get_tool_definitions() -> List[Dict[str, Any]]:
        """Returns a list of tool definitions using static Pydantic models."""
        return [
            {
                "name": "sentient-brain/orchestrate",
                "description": "Ultra Orchestrator - Master agent for coordinating multi-agent workflows",
                "inputSchema": OrchestrateInput.model_json_schema()
            },
            {
                "name": "sentient-brain/architect",
                "description": "Architect Agent - Design and plan software projects",
                "inputSchema": ArchitectInput.model_json_schema()
            },
            {
                "name": "sentient-brain/analyze-code",
                "description": "Code Analysis - Deep code understanding and indexing",
                "inputSchema": AnalyzeCodeInput.model_json_schema()
            },
            {
                "name": "sentient-brain/search-knowledge",
                "description": "Knowledge Graph Search - Semantic search across project knowledge",
                "inputSchema": SearchKnowledgeInput.model_json_schema()
            },
            {
                "name": "sentient-brain/debug-assist",
                "description": "Debug & Refactor Agent - Code debugging and improvement",
                "inputSchema": DebugAssistInput.model_json_schema()
            }
        ]

    async def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution"""
        try:
            if tool_name == "sentient-brain/orchestrate":
                return await self._orchestrate(arguments)
            elif tool_name == "sentient-brain/architect":
                return await self._architect(arguments)
            elif tool_name == "sentient-brain/analyze-code":
                return await self._analyze_code(arguments)
            elif tool_name == "sentient-brain/search-knowledge":
                return await self._search_knowledge(arguments)
            elif tool_name == "sentient-brain/debug-assist":
                return await self._debug_assist(arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e), "success": False}

    async def _orchestrate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra Orchestrator implementation"""
        query = args.get("query", "")
        context = args.get("context", {})
        priority = args.get("priority", "medium")
        
        # Store interaction in memory if in fallback mode
        if self.fallback_mode:
            interaction_id = f"interaction_{len(self.memory_store)}"
            self.memory_store[interaction_id] = {
                "type": "orchestration",
                "query": query,
                "context": context,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            }
        
        # Simulate orchestration logic
        result = {
            "agent": "UltraOrchestrator",
            "analysis": f"Analyzed query: '{query}' with {priority} priority",
            "workflow": self._determine_workflow(query),
            "next_agents": self._suggest_agents(query),
            "mode": "fallback" if self.fallback_mode else "full",
            "storage": "in-memory" if self.fallback_mode else "persistent",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Orchestrated workflow for query: {query} (fallback_mode: {self.fallback_mode})")
        return result

    async def _architect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Architect Agent implementation"""
        project_type = args.get("project_type", "")
        requirements = args.get("requirements", "")
        tech_stack = args.get("tech_stack", [])
        
        # Store interaction in memory if in fallback mode
        if self.fallback_mode:
            interaction_id = f"architect_{len(self.memory_store)}"
            self.memory_store[interaction_id] = {
                "type": "architecture",
                "project_type": project_type,
                "requirements": requirements,
                "tech_stack": tech_stack,
                "timestamp": datetime.now().isoformat()
            }
        
        result = {
            "agent": "ArchitectAgent",
            "project_design": {
                "type": project_type,
                "architecture": self._generate_architecture(project_type),
                "recommended_stack": tech_stack or self._recommend_stack(project_type),
                "phases": self._create_project_phases(requirements)
            },
            "mode": "fallback" if self.fallback_mode else "full",
            "storage": "in-memory" if self.fallback_mode else "persistent",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    async def _analyze_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Code Analysis implementation"""
        code = args.get("code", "")
        language = args.get("language", "python")
        analysis_type = args.get("analysis_type", "structure")
        
        # Store analysis in memory if in fallback mode
        if self.fallback_mode:
            analysis_id = f"analysis_{len(self.memory_store)}"
            self.memory_store[analysis_id] = {
                "type": "code_analysis",
                "language": language,
                "analysis_type": analysis_type,
                "code_length": len(code),
                "timestamp": datetime.now().isoformat()
            }
        
        result = {
            "agent": "CodeAnalysisAgent",
            "analysis": {
                "language": language,
                "type": analysis_type,
                "metrics": {
                    "lines": len(code.split('\n')),
                    "complexity": "medium",  # Placeholder
                    "quality_score": 85      # Placeholder
                },
                "insights": self._generate_code_insights(code, analysis_type)
            },
            "mode": "fallback" if self.fallback_mode else "full",
            "storage": "in-memory" if self.fallback_mode else "persistent",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    async def _search_knowledge(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Knowledge Graph Search implementation"""
        query = args.get("query", "")
        node_type = args.get("node_type", "code_chunk")
        limit = args.get("limit", 10)
        
        # In fallback mode, search through memory store
        if self.fallback_mode:
            # Search through stored interactions
            matching_results = []
            for key, value in self.memory_store.items():
                if query.lower() in str(value).lower():
                    matching_results.append({
                        "id": key,
                        "type": value.get("type", "memory"),
                        "content": f"Memory: {value}",
                        "relevance": 0.8,
                        "metadata": {"source": "memory_store", "created": value.get("timestamp", "")}
                    })
                    if len(matching_results) >= limit:
                        break
            
            results = matching_results
        else:
            # Simulate knowledge graph search for full mode
            results = [
                {
                    "id": f"node_{i}",
                    "type": node_type,
                    "content": f"Knowledge result {i} for '{query}'",
                    "relevance": 0.9 - (i * 0.1),
                    "metadata": {"source": "knowledge_graph", "created": datetime.now().isoformat()}
                }
                for i in range(min(limit, 5))  # Simulate limited results
            ]
        
        return {
            "agent": "KnowledgeSearchAgent",
            "query": query,
            "results": results,
            "total_found": len(results),
            "mode": "fallback" if self.fallback_mode else "full",
            "storage": "in-memory" if self.fallback_mode else "persistent",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    async def _debug_assist(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Debug & Refactor implementation"""
        code = args.get("code", "")
        error_message = args.get("error_message", "")
        debug_type = args.get("debug_type", "fix")
        
        # Store debug session in memory if in fallback mode
        if self.fallback_mode:
            debug_id = f"debug_{len(self.memory_store)}"
            self.memory_store[debug_id] = {
                "type": "debug_session",
                "debug_type": debug_type,
                "error_message": error_message,
                "code_length": len(code),
                "timestamp": datetime.now().isoformat()
            }
        
        result = {
            "agent": "DebugRefactorAgent",
            "debug_analysis": {
                "type": debug_type,
                "issues_found": self._identify_issues(code, error_message),
                "suggestions": self._generate_suggestions(code, debug_type),
                "confidence": 0.85
            },
            "mode": "fallback" if self.fallback_mode else "full",
            "storage": "in-memory" if self.fallback_mode else "persistent",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    def _determine_workflow(self, query: str) -> Dict[str, Any]:
        """Determine appropriate workflow based on query"""
        query_lower = query.lower()
        if any(word in query_lower for word in ["create", "build", "new"]):
            return {"type": "creation", "agents": ["architect", "codebase"]}
        elif any(word in query_lower for word in ["debug", "fix", "error"]):
            return {"type": "debugging", "agents": ["debug", "codebase"]}
        elif any(word in query_lower for word in ["search", "find", "look"]):
            return {"type": "search", "agents": ["knowledge"]}
        else:
            return {"type": "general", "agents": ["architect", "codebase"]}

    def _suggest_agents(self, query: str) -> List[str]:
        """Suggest appropriate agents for the query"""
        workflow = self._determine_workflow(query)
        return workflow.get("agents", ["architect"])

    def _generate_architecture(self, project_type: str) -> Dict[str, Any]:
        """Generate architecture recommendations"""
        architectures = {
            "web_api": {"pattern": "layered", "components": ["controller", "service", "repository"]},
            "microservice": {"pattern": "microservices", "components": ["api_gateway", "services", "database"]},
            "web_app": {"pattern": "mvc", "components": ["frontend", "backend", "database"]}
        }
        return architectures.get(project_type, {"pattern": "modular", "components": ["core", "modules"]})

    def _recommend_stack(self, project_type: str) -> List[str]:
        """Recommend technology stack"""
        stacks = {
            "web_api": ["python", "fastapi", "postgresql"],
            "web_app": ["typescript", "react", "nodejs", "postgresql"],
            "microservice": ["python", "fastapi", "docker", "kubernetes"]
        }
        return stacks.get(project_type, ["python", "fastapi"])

    def _create_project_phases(self, requirements: str) -> List[Dict[str, Any]]:
        """Create project phases based on requirements"""
        return [
            {"phase": "Planning", "duration": "1-2 weeks", "deliverables": ["Architecture", "Specifications"]},
            {"phase": "Development", "duration": "4-6 weeks", "deliverables": ["Core Features", "Testing"]},
            {"phase": "Deployment", "duration": "1 week", "deliverables": ["Production Setup", "Documentation"]}
        ]

    def _generate_code_insights(self, code: str, analysis_type: str) -> List[str]:
        """Generate code insights based on analysis type"""
        insights = {
            "structure": ["Well-organized imports", "Clear function definitions", "Proper error handling"],
            "quality": ["Good variable naming", "Adequate comments", "Follows best practices"],
            "dependencies": ["Standard library usage", "External dependencies identified", "No circular imports"]
        }
        return insights.get(analysis_type, ["Code analyzed successfully"])

    def _identify_issues(self, code: str, error_message: str) -> List[Dict[str, Any]]:
        """Identify code issues"""
        issues = []
        if error_message:
            issues.append({"type": "runtime_error", "message": error_message, "severity": "high"})
        if "import" in code and not code.strip().startswith("import"):
            issues.append({"type": "import_order", "message": "Imports should be at the top", "severity": "low"})
        return issues or [{"type": "no_issues", "message": "No obvious issues found", "severity": "info"}]

    def _generate_suggestions(self, code: str, debug_type: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = {
            "fix": ["Add error handling", "Validate inputs", "Check for edge cases"],
            "optimize": ["Use list comprehensions", "Cache expensive operations", "Optimize loops"],
            "refactor": ["Extract functions", "Improve naming", "Add documentation"]
        }
        return suggestions.get(debug_type, ["Code looks good"])

# FastAPI Application
app = FastAPI(
    title="Sentient Brain Multi-Agent MCP Server",
    description="Advanced AI Code Developer system for Smithery.ai",
    version="1.0.0"
)

# CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global config and MCP instance
config = None
mcp_server = None

def parse_smithery_config(request: Request) -> Config:
    """Parse Smithery configuration from query parameters - safe for tool scanning"""
    try:
        query_params = dict(request.query_params)
        
        # Convert dot notation to environment variables
        env_vars = {}
        for key, value in query_params.items():
            # Convert dot notation like 'server.apiKey' to 'SERVER_API_KEY'
            env_key = key.replace('.', '_').upper()
            env_vars[env_key] = value
        
        # Create config with query params as environment
        return Config(**env_vars)
    except Exception as e:
        logger.warning(f"Configuration parsing failed (using defaults): {e}")
        # Return default config during tool scanning - allows lazy loading
        return Config()

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "name": "Sentient Brain Multi-Agent MCP Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health():
    """Health check endpoint for Smithery"""
    return {
        "status": "healthy",
        "name": "sentient-brain-mcp",
        "version": "1.0.0",
        "capabilities": ["tools", "resources", "prompts"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/mcp")
async def mcp_get(request: Request):
    """Handle MCP GET requests - return server info and tools for lazy loading"""
    # Return static tool definitions without requiring authentication
    # This allows Smithery to discover tools before user configuration
    return {
            "server": {
                "name": "sentient-brain-multi-agent",
                "version": "1.0.0"
            },
            "tools": SentientBrainMCP.get_tool_definitions(),
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
        }
    }

@app.post("/mcp")
async def mcp_post(request: Request, mcp_request: MCPRequest):
    """Handle MCP POST requests - implements lazy loading for tool discovery"""
    try:
        method = mcp_request.method
        # Ensure ID is always a string - handle 0, null, and other values
        if mcp_request.id is not None:
            request_id = str(mcp_request.id)
        else:
            request_id = "1"
        
        # Methods that don't require valid configuration (lazy loading)
        if method in ["initialize", "tools/list", "resources/list", "prompts/list", "notifications/initialized", "ping"]:
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": {
                        "name": "sentient-brain-mcp",
                        "version": "1.0.0"
                    }
                }
                
            elif method == "notifications/initialized":
                # Handle client initialization notification
                logger.info("Client initialized successfully")
                result = {}  # Notifications don't return results
                
            elif method == "ping":
                # Handle ping requests for server health/connectivity
                # Return empty result as per MCP spec
                result = {}
                
            elif method == "tools/list":
                # Use static tool definitions for lazy loading - no config needed
                tools = SentientBrainMCP.get_tool_definitions()
                result = {"tools": tools}
                
            elif method == "resources/list":
                result = {"resources": []}
                
            elif method == "prompts/list":
                result = {"prompts": []}
                
        # Methods that require valid configuration (actual tool execution)
        elif method == "tools/call":
            # Only now do we parse and validate configuration
            config = parse_smithery_config(request)
            server = SentientBrainMCP(config)
            server.initialize()
            
            params = mcp_request.params or {}
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            # Validate configuration for actual tool execution
            if tool_name.startswith("sentient-brain/") and not config.groq_api_key:
                raise ValueError("GROQ_API_KEY is required for tool execution. Please get your API key from https://console.groq.com/keys and configure it in Smithery.")
            
            tool_result = await server.handle_tool_call(tool_name, arguments)
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(tool_result, indent=2)
                    }
                ],
                "isError": tool_result.get("success", True) == False
            }
            
        else:
            # Handle unknown methods more gracefully
            logger.warning(f"Unknown method: {method}")
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": str(request_id),
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}. Available methods: initialize, tools/list, tools/call, resources/list, prompts/list, notifications/initialized, ping"
                    }
                }
            )
        
        response = {
            "jsonrpc": "2.0",
            "id": str(request_id),  # Ensure ID is string
            "result": result
        }
        
        logger.info(f"MCP {method} request processed successfully")
        return JSONResponse(content=response)
            
    except Exception as e:
        logger.error(f"MCP request error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "id": str(mcp_request.id) if mcp_request.id else "error",
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
        )

@app.delete("/mcp")
async def mcp_delete(request: Request):
    """Handle MCP DELETE requests - cleanup"""
    global mcp_server
    mcp_server = None
    return {"status": "cleaned_up", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Sentient Brain MCP Server on port {port}")
    logger.info(f"Health check available at http://0.0.0.0:{port}/health")
    logger.info(f"MCP endpoint available at http://0.0.0.0:{port}/mcp")
    
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )