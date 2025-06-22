#!/usr/bin/env python3
"""
Sentient Brain Multi-Agent MCP Server for Smithery.ai
Optimized for HTTP streaming protocol with configuration via query parameters
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
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

class Config(BaseSettings):
    """Configuration loaded from environment or Smithery query params"""
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    surreal_url: str = Field(default="ws://localhost:8000/rpc", alias="SURREAL_URL")
    surreal_user: str = Field(default="root", alias="SURREAL_USER")
    surreal_pass: str = Field(default="root", alias="SURREAL_PASS")
    surreal_namespace: str = Field(default="sentient_brain", alias="SURREAL_NAMESPACE")
    surreal_database: str = Field(default="multi_agent", alias="SURREAL_DATABASE")
    groq_model: str = Field(default="llama-3.1-70b-versatile", alias="GROQ_MODEL")
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    port: int = Field(default=8000, alias="PORT")

    class Config:
        env_file = ".env"
        case_sensitive = False

class MCPRequest(BaseModel):
    """MCP Protocol Request"""
    jsonrpc: str = "2.0"
    id: Optional[str] = None
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
        self.tools = self._initialize_tools()
        self.agents = {}
        logger.info(f"Initialized Sentient Brain MCP with {len(self.tools)} tools")

    def _initialize_tools(self) -> List[Dict[str, Any]]:
        """Initialize available MCP tools"""
        return [
            {
                "name": "sentient-brain/orchestrate",
                "description": "Ultra Orchestrator - Master agent for coordinating multi-agent workflows",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "User query or task description"},
                        "context": {"type": "object", "description": "Additional context"},
                        "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "sentient-brain/architect",
                "description": "Architect Agent - Design and plan software projects",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_type": {"type": "string", "description": "Type of project to architect"},
                        "requirements": {"type": "string", "description": "Project requirements"},
                        "tech_stack": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["project_type", "requirements"]
                }
            },
            {
                "name": "sentient-brain/analyze-code",
                "description": "Code Analysis - Deep code understanding and indexing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to analyze"},
                        "language": {"type": "string", "description": "Programming language"},
                        "analysis_type": {"type": "string", "enum": ["structure", "quality", "dependencies"]}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "sentient-brain/search-knowledge",
                "description": "Knowledge Graph Search - Semantic search across project knowledge",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "node_type": {"type": "string", "enum": ["code_chunk", "task", "document", "concept"]},
                        "limit": {"type": "integer", "default": 10}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "sentient-brain/debug-assist",
                "description": "Debug & Refactor Agent - Code debugging and improvement",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code with issues"},
                        "error_message": {"type": "string", "description": "Error message if available"},
                        "debug_type": {"type": "string", "enum": ["fix", "optimize", "refactor"]}
                    },
                    "required": ["code"]
                }
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
        
        # Simulate orchestration logic
        result = {
            "agent": "UltraOrchestrator",
            "analysis": f"Analyzed query: '{query}' with {priority} priority",
            "workflow": self._determine_workflow(query),
            "next_agents": self._suggest_agents(query),
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Orchestrated workflow for query: {query}")
        return result

    async def _architect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Architect Agent implementation"""
        project_type = args.get("project_type", "")
        requirements = args.get("requirements", "")
        tech_stack = args.get("tech_stack", [])
        
        result = {
            "agent": "ArchitectAgent",
            "project_design": {
                "type": project_type,
                "architecture": self._generate_architecture(project_type),
                "recommended_stack": tech_stack or self._recommend_stack(project_type),
                "phases": self._create_project_phases(requirements)
            },
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    async def _analyze_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Code Analysis implementation"""
        code = args.get("code", "")
        language = args.get("language", "python")
        analysis_type = args.get("analysis_type", "structure")
        
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
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        return result

    async def _search_knowledge(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Knowledge Graph Search implementation"""
        query = args.get("query", "")
        node_type = args.get("node_type", "code_chunk")
        limit = args.get("limit", 10)
        
        # Simulate knowledge graph search
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
            "success": True,
            "timestamp": datetime.now().isoformat()
        }

    async def _debug_assist(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Debug & Refactor implementation"""
        code = args.get("code", "")
        error_message = args.get("error_message", "")
        debug_type = args.get("debug_type", "fix")
        
        result = {
            "agent": "DebugRefactorAgent",
            "debug_analysis": {
                "type": debug_type,
                "issues_found": self._identify_issues(code, error_message),
                "suggestions": self._generate_suggestions(code, debug_type),
                "confidence": 0.85
            },
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
    """Parse Smithery configuration from query parameters"""
    query_params = dict(request.query_params)
    
    # Convert dot notation to environment variables
    env_vars = {}
    for key, value in query_params.items():
        # Convert dot notation like 'server.apiKey' to 'SERVER_API_KEY'
        env_key = key.replace('.', '_').upper()
        env_vars[env_key] = value
    
    # Create config with query params as environment
    return Config(**env_vars)

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "name": "Sentient Brain Multi-Agent MCP Server",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/mcp")
async def mcp_get(request: Request):
    """Handle MCP GET requests - return server info and tools"""
    global config, mcp_server
    
    # Initialize with Smithery config
    config = parse_smithery_config(request)
    mcp_server = SentientBrainMCP(config)
    
    return {
        "jsonrpc": "2.0",
        "result": {
            "server": {
                "name": "sentient-brain-multi-agent",
                "version": "1.0.0"
            },
            "tools": mcp_server.tools,
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True
            }
        }
    }

@app.post("/mcp")
async def mcp_post(request: Request, mcp_request: MCPRequest):
    """Handle MCP POST requests - execute tools"""
    global config, mcp_server
    
    if not mcp_server:
        config = parse_smithery_config(request)
        mcp_server = SentientBrainMCP(config)
    
    try:
        if mcp_request.method == "tools/call":
            params = mcp_request.params or {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            
            if not tool_name:
                raise HTTPException(status_code=400, detail="Tool name is required")
            
            result = await mcp_server.handle_tool_call(tool_name, arguments)
            
            return MCPResponse(
                id=mcp_request.id,
                result={
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2)
                        }
                    ]
                }
            )
        
        elif mcp_request.method == "tools/list":
            return MCPResponse(
                id=mcp_request.id,
                result={"tools": mcp_server.tools}
            )
        
        else:
            return MCPResponse(
                id=mcp_request.id,
                error={"code": -32601, "message": f"Method not found: {mcp_request.method}"}
            )
            
    except Exception as e:
        logger.error(f"MCP request error: {e}")
        return MCPResponse(
            id=mcp_request.id,
            error={"code": -32603, "message": f"Internal error: {str(e)}"}
        )

@app.delete("/mcp")
async def mcp_delete(request: Request):
    """Handle MCP DELETE requests - cleanup"""
    global mcp_server
    mcp_server = None
    return {"status": "cleaned_up", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "mcp_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )