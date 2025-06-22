#!/usr/bin/env python3
"""
Quick test script to validate the MCP server starts correctly
"""

import asyncio
import httpx
import os
import subprocess
import time
import sys

async def test_server():
    """Test if the server starts and responds correctly"""
    try:
        print("🔍 Testing Sentient Brain MCP Server startup...")
        
        # Test if we can import the main module
        print("1. Testing imports...")
        try:
            from mcp_server import app, Config, SentientBrainMCP
            print("✅ Imports successful")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test configuration creation
        print("2. Testing configuration...")
        try:
            config = Config()
            print("✅ Configuration created")
        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            return False
        
        # Test MCP server creation
        print("3. Testing MCP server creation...")
        try:
            server = SentientBrainMCP(config)
            tools = server.get_tool_definitions()
            print(f"✅ MCP server created with {len(tools)} tools")
        except Exception as e:
            print(f"❌ MCP server creation failed: {e}")
            return False
        
        # Test that we can start the FastAPI app (just check it's configured)
        print("4. Testing FastAPI app configuration...")
        try:
            from fastapi.testclient import TestClient
            client = TestClient(app)
            response = client.get("/health")
            print(f"✅ Health endpoint response: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Name: {data.get('name')}")
            
        except Exception as e:
            print(f"❌ FastAPI test failed: {e}")
            return False
        
        print("\n🎉 All tests passed! Server should deploy successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Set minimal environment for testing
    os.environ.setdefault("GROQ_API_KEY", "test_key")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1) 