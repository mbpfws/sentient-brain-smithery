#!/usr/bin/env python3
"""
Test script to verify the server starts correctly and handles tool scanning
"""
import asyncio
import json
import httpx
import time
import subprocess
import os
import signal

async def test_server_startup():
    """Test that the server starts and responds correctly"""
    print("Testing Sentient Brain MCP Server startup...")
    
    # Start the server in background
    server_process = subprocess.Popen([
        "python", "mcp_server.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server to start
        await asyncio.sleep(3)
        
        # Test health endpoint
        print("\n--- Testing Health Endpoint ---")
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get("http://localhost:8000/health")
                print(f"Health Status: {response.status_code}")
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"Health Response: {health_data}")
                else:
                    print(f"Health Error: {response.text}")
            except Exception as e:
                print(f"Health check failed: {e}")
        
        # Test MCP GET endpoint (tool scanning)
        print("\n--- Testing Tool Scanning (MCP GET) ---")
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get("http://localhost:8000/mcp")
                print(f"MCP GET Status: {response.status_code}")
                if response.status_code == 200:
                    mcp_data = response.json()
                    tools = mcp_data.get("tools", [])
                    print(f"Tools found: {len(tools)}")
                    for tool in tools:
                        print(f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                else:
                    print(f"MCP GET Error: {response.text}")
            except Exception as e:
                print(f"Tool scanning failed: {e}")
        
        # Test MCP POST tools/list
        print("\n--- Testing MCP POST tools/list ---")
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                mcp_request = {
                    "jsonrpc": "2.0",
                    "id": "test",
                    "method": "tools/list",
                    "params": {}
                }
                response = await client.post(
                    "http://localhost:8000/mcp",
                    json=mcp_request,
                    headers={"Content-Type": "application/json"}
                )
                print(f"MCP POST Status: {response.status_code}")
                if response.status_code == 200:
                    mcp_response = response.json()
                    tools = mcp_response.get("result", {}).get("tools", [])
                    print(f"Tools via POST: {len(tools)}")
                else:
                    print(f"MCP POST Error: {response.text}")
            except Exception as e:
                print(f"MCP POST failed: {e}")
        
        print("\n✅ Server startup test completed!")
        
    finally:
        # Clean up server process
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except:
            server_process.kill()
        print("Server process terminated")

if __name__ == "__main__":
    asyncio.run(test_server_startup()) 