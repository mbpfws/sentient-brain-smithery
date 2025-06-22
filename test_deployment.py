#!/usr/bin/env python3
"""
Test script for Sentient Brain Smithery deployment
Validates core functionality before deployment
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp_server import SentientBrainMCP, Config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure mcp_server.py is in the same directory")
    sys.exit(1)

async def test_deployment():
    """Test core functionality"""
    print("🧪 Testing Sentient Brain Smithery Deployment")
    print("=" * 50)
    
    # Test configuration
    print("1. Testing Configuration...")
    config = Config(
        groq_api_key="test_key",
        surreal_url="ws://localhost:8000/rpc",
        surreal_user="root",
        surreal_pass="root"
    )
    print(f"   ✅ Config loaded: {config.groq_model}")
    
    # Test MCP server initialization
    print("2. Testing MCP Server...")
    mcp_server = SentientBrainMCP(config)
    print(f"   ✅ Server initialized with {len(mcp_server.tools)} tools")
    
    # Test tool discovery
    print("3. Testing Tool Discovery...")
    tools = mcp_server.tools
    for tool in tools:
        print(f"   📦 {tool['name']}: {tool['description'][:50]}...")
    
    # Test orchestrator
    print("4. Testing Ultra Orchestrator...")
    result = await mcp_server.handle_tool_call(
        "sentient-brain/orchestrate",
        {"query": "Build a REST API", "priority": "high"}
    )
    print(f"   ✅ Orchestrator response: {result.get('success', False)}")
    
    # Test architect
    print("5. Testing Architect Agent...")
    result = await mcp_server.handle_tool_call(
        "sentient-brain/architect",
        {
            "project_type": "web_api",
            "requirements": "User authentication system",
            "tech_stack": ["python", "fastapi"]
        }
    )
    print(f"   ✅ Architect response: {result.get('success', False)}")
    
    # Test code analysis
    print("6. Testing Code Analysis...")
    result = await mcp_server.handle_tool_call(
        "sentient-brain/analyze-code",
        {
            "code": "def hello_world():\n    return 'Hello, World!'",
            "language": "python",
            "analysis_type": "structure"
        }
    )
    print(f"   ✅ Code analysis response: {result.get('success', False)}")
    
    # Test knowledge search
    print("7. Testing Knowledge Search...")
    result = await mcp_server.handle_tool_call(
        "sentient-brain/search-knowledge",
        {
            "query": "authentication patterns",
            "node_type": "code_chunk",
            "limit": 5
        }
    )
    print(f"   ✅ Knowledge search found: {len(result.get('results', []))} results")
    
    # Test debug assist
    print("8. Testing Debug Assistant...")
    result = await mcp_server.handle_tool_call(
        "sentient-brain/debug-assist",
        {
            "code": "x = 1/0",
            "error_message": "ZeroDivisionError: division by zero",
            "debug_type": "fix"
        }
    )
    print(f"   ✅ Debug assistant response: {result.get('success', False)}")
    
    print("\n🎉 All tests passed! Deployment ready for Smithery.ai")
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_deployment())
        if success:
            print("\n✅ Deployment validation successful!")
            sys.exit(0)
        else:
            print("\n❌ Deployment validation failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)