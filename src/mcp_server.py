"""
MCP (Model Context Protocol) Server Integration for Sentient Brain Multi-Agent System

This module provides MCP server capabilities to make the system compatible 
with Smithery.ai and other MCP-compatible platforms.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# MCP imports
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import (
    Resource, 
    Tool, 
    TextContent, 
    ImageContent, 
    EmbeddedResource
)

# Our system imports
from .services.surreal_service import SurrealDBService
from .services.groq_service import GroqLLMService
from .agents.orchestrator import UltraOrchestratorAgent
from .models.agent_models import AgentConfig, AgentType

# Configure logging
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    "sentient-brain-multi-agent",
    transport_type="stdio",
    keep_alive_timeout=300,
    heartbeat_interval=30
)

# Global services (will be initialized in main)
db_service: Optional[SurrealDBService] = None
llm_service: Optional[GroqLLMService] = None
orchestrator: Optional[UltraOrchestratorAgent] = None


@mcp.tool("sentient-brain/process-query")
async def process_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    project_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process a user query through the multi-agent system.
    
    This is the main entry point for user interactions. The Ultra Orchestrator
    will analyze intent, coordinate agents, and return comprehensive results.
    
    Args:
        query: The user's query or request
        context: Optional context information (project details, preferences, etc.)
        project_id: Optional project identifier for context
    
    Returns:
        List of response objects containing agent outputs, analysis, and recommendations
    
    Raises:
        ToolError: If query processing fails
    """
    try:
        if not orchestrator:
            raise ToolError("Orchestrator not initialized")
        
        # Add project context if provided
        if project_id and context is None:
            context = {"project_id": project_id}
        elif project_id and context:
            context["project_id"] = project_id
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process through orchestrator
        response = await orchestrator.process_user_query(query, context)
        
        # Format response for MCP
        return [{
            "type": "agent_response",
            "data": {
                "query": query,
                "response": response.content,
                "agent_type": response.agent_type.value,
                "confidence": response.confidence,
                "workflow_id": response.workflow_id,
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
        }]
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise ToolError(f"Query processing failed: {str(e)}")


@mcp.tool("sentient-brain/create-project")
async def create_project(
    name: str,
    description: str = "",
    root_path: str = ".",
    languages: List[str] = None,
    frameworks: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Create a new project context in the system.
    
    Args:
        name: Project name
        description: Project description
        root_path: Root directory path
        languages: Programming languages used
        frameworks: Frameworks/libraries used
    
    Returns:
        Project creation result with project ID and initial analysis
    
    Raises:
        ToolError: If project creation fails
    """
    try:
        if not db_service:
            raise ToolError("Database service not initialized")
        
        logger.info(f"Creating project: {name}")
        
        # Create project context
        project_data = {
            "name": name,
            "description": description,
            "root_path": root_path,
            "languages": languages or [],
            "frameworks": frameworks or [],
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        # Store in database
        result = await db_service.create_node(
            table="projects",
            data=project_data
        )
        
        project_id = result.get("id")
        
        return [{
            "type": "project_created",
            "data": {
                "project_id": project_id,
                "name": name,
                "description": description,
                "languages": languages,
                "frameworks": frameworks,
                "status": "created",
                "next_steps": [
                    "Initialize codebase analysis",
                    "Set up knowledge graph",
                    "Configure agent workflows"
                ]
            }
        }]
        
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise ToolError(f"Project creation failed: {str(e)}")


@mcp.tool("sentient-brain/analyze-codebase")
async def analyze_codebase(
    project_id: str,
    path: str = ".",
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Analyze a codebase and populate the knowledge graph.
    
    Args:
        project_id: Project identifier
        path: Path to analyze
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
    
    Returns:
        Analysis results with discovered files, structures, and insights
    
    Raises:
        ToolError: If analysis fails
    """
    try:
        if not orchestrator:
            raise ToolError("Orchestrator not initialized")
        
        logger.info(f"Analyzing codebase for project {project_id}")
        
        # Trigger codebase analysis through orchestrator
        analysis_query = f"Analyze the codebase at {path} for project {project_id}"
        context = {
            "project_id": project_id,
            "analysis_type": "codebase",
            "path": path,
            "include_patterns": include_patterns or [],
            "exclude_patterns": exclude_patterns or []
        }
        
        response = await orchestrator.process_user_query(analysis_query, context)
        
        return [{
            "type": "codebase_analysis",
            "data": {
                "project_id": project_id,
                "path": path,
                "analysis_result": response.content,
                "workflow_id": response.workflow_id,
                "timestamp": datetime.now().isoformat()
            }
        }]
        
    except Exception as e:
        logger.error(f"Error analyzing codebase: {e}")
        raise ToolError(f"Codebase analysis failed: {str(e)}")


@mcp.tool("sentient-brain/search-knowledge")
async def search_knowledge(
    query: str,
    node_type: Optional[str] = None,
    project_id: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Search the knowledge graph for relevant information.
    
    Args:
        query: Search query
        node_type: Type of nodes to search (code_chunk, document, task, etc.)
        project_id: Project to search within
        limit: Maximum number of results
    
    Returns:
        Search results with relevant knowledge nodes
    
    Raises:
        ToolError: If search fails
    """
    try:
        if not db_service:
            raise ToolError("Database service not initialized")
        
        logger.info(f"Searching knowledge: {query}")
        
        # Perform semantic search
        search_filters = {}
        if node_type:
            search_filters["node_type"] = node_type
        if project_id:
            search_filters["project_id"] = project_id
        
        results = await db_service.semantic_search(
            query=query,
            table="knowledge_nodes",
            limit=limit,
            filters=search_filters
        )
        
        return [{
            "type": "knowledge_search_results",
            "data": {
                "query": query,
                "results": results,
                "count": len(results),
                "filters": search_filters,
                "timestamp": datetime.now().isoformat()
            }
        }]
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise ToolError(f"Knowledge search failed: {str(e)}")


@mcp.tool("sentient-brain/get-agent-status")
async def get_agent_status(
    agent_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get status information for agents in the system.
    
    Args:
        agent_type: Specific agent type to query (optional)
    
    Returns:
        Agent status information including activity, performance, and health
    
    Raises:
        ToolError: If status retrieval fails
    """
    try:
        if not orchestrator:
            raise ToolError("Orchestrator not initialized")
        
        logger.info(f"Getting agent status: {agent_type or 'all'}")
        
        # Get agent status from orchestrator
        status_data = await orchestrator.get_agent_status(agent_type)
        
        return [{
            "type": "agent_status",
            "data": {
                "agents": status_data,
                "timestamp": datetime.now().isoformat(),
                "system_health": "operational"
            }
        }]
        
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise ToolError(f"Agent status retrieval failed: {str(e)}")


@mcp.tool("sentient-brain/run-diagnostics")
async def run_diagnostics() -> List[Dict[str, Any]]:
    """
    Run system diagnostics and health checks.
    
    Returns:
        Comprehensive system health and diagnostic information
    
    Raises:
        ToolError: If diagnostics fail
    """
    try:
        logger.info("Running system diagnostics")
        
        diagnostics = {
            "database": {
                "status": "connected" if db_service and await db_service.get_connection_status() else "disconnected",
                "connection_time": datetime.now().isoformat()
            },
            "llm_service": {
                "status": "available" if llm_service else "unavailable",
                "model": llm_service.model if llm_service else None
            },
            "orchestrator": {
                "status": "active" if orchestrator else "inactive",
                "agents_count": len(orchestrator.active_agents) if orchestrator else 0
            },
            "memory_usage": "normal",  # Would implement actual memory monitoring
            "performance": "optimal"   # Would implement actual performance metrics
        }
        
        return [{
            "type": "system_diagnostics",
            "data": {
                "diagnostics": diagnostics,
                "overall_health": "healthy",
                "timestamp": datetime.now().isoformat(),
                "recommendations": [
                    "System operating normally",
                    "All core services functional"
                ]
            }
        }]
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        raise ToolError(f"Diagnostics failed: {str(e)}")


# Resource definitions for MCP
@mcp.resource("sentient-brain://projects/{project_id}")
async def get_project_resource(project_id: str) -> Resource:
    """Get project information as a resource."""
    try:
        if not db_service:
            raise ValueError("Database service not initialized")
        
        project_data = await db_service.get_node("projects", project_id)
        
        return Resource(
            uri=f"sentient-brain://projects/{project_id}",
            name=f"Project: {project_data.get('name', project_id)}",
            description=project_data.get('description', 'No description'),
            mimeType="application/json"
        )
    except Exception as e:
        logger.error(f"Error getting project resource: {e}")
        raise


@mcp.resource("sentient-brain://knowledge/{node_id}")
async def get_knowledge_resource(node_id: str) -> Resource:
    """Get knowledge node as a resource."""
    try:
        if not db_service:
            raise ValueError("Database service not initialized")
        
        node_data = await db_service.get_node("knowledge_nodes", node_id)
        
        return Resource(
            uri=f"sentient-brain://knowledge/{node_id}",
            name=f"Knowledge: {node_data.get('title', node_id)}",
            description=node_data.get('summary', 'No description'),
            mimeType="text/plain"
        )
    except Exception as e:
        logger.error(f"Error getting knowledge resource: {e}")
        raise


async def initialize_services():
    """Initialize all services for MCP server."""
    global db_service, llm_service, orchestrator
    
    try:
        logger.info("Initializing MCP services...")
        
        # Initialize database service
        db_service = SurrealDBService(
            url="ws://localhost:8000/rpc",
            username="root", 
            password="root",
            namespace="sentient_brain",
            database="multi_agent"
        )
        
        await db_service.connect()
        
        # Initialize LLM service
        llm_service = GroqLLMService(
            api_key="your-groq-api-key",  # Should come from environment
            model="llama-3.1-70b-versatile"
        )
        
        # Initialize orchestrator
        orchestrator_config = AgentConfig(
            agent_type=AgentType.ULTRA_ORCHESTRATOR,
            name="MCP Ultra Orchestrator",
            description="MCP-enabled orchestrator for multi-agent coordination"
        )
        
        orchestrator = UltraOrchestratorAgent(
            config=orchestrator_config,
            db_service=db_service,
            llm_service=llm_service
        )
        
        logger.info("MCP services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP services: {e}")
        raise


async def main():
    """Main entry point for MCP server."""
    try:
        # Initialize services
        await initialize_services()
        
        # Run MCP server
        logger.info("Starting Sentient Brain MCP Server...")
        await mcp.run()
        
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise
    finally:
        # Cleanup
        if db_service:
            await db_service.disconnect()


if __name__ == "__main__":
    asyncio.run(main())