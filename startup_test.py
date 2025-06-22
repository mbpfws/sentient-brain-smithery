#!/usr/bin/env python3
"""
Simple startup test for container debugging
"""
import os
import sys
import time
import requests
from pathlib import Path

def test_startup():
    """Test if the server can start and respond"""
    print("🔍 Startup Test - Container Environment")
    print("=" * 50)
    
    # 1. Check Python environment
    print(f"✅ Python version: {sys.version}")
    print(f"✅ Working directory: {os.getcwd()}")
    print(f"✅ Python path: {sys.path}")
    
    # 2. Check required files
    required_files = ["mcp_server.py", "requirements.txt", "smithery.yaml"]
    for file in required_files:
        if Path(file).exists():
            print(f"✅ Found {file}")
        else:
            print(f"❌ Missing {file}")
            return False
    
    # 3. Test imports
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Core dependencies available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 4. Test app creation
    try:
        from mcp_server import app
        print("✅ FastAPI app created successfully")
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False
    
    # 5. Check environment variables
    port = os.getenv("PORT", "8000")
    print(f"✅ Port configured: {port}")
    
    # 6. Test server startup (quick test)
    print("\n🚀 Testing server startup...")
    try:
        import subprocess
        import signal
        
        # Start server in background
        proc = subprocess.Popen([
            "python", "mcp_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Test health endpoint
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Health endpoint responding")
                data = response.json()
                print(f"   Status: {data.get('status')}")
                print(f"   Name: {data.get('name')}")
            else:
                print(f"❌ Health endpoint returned {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Health endpoint not accessible: {e}")
        
        # Stop the server
        proc.terminate()
        proc.wait(timeout=5)
        
    except Exception as e:
        print(f"❌ Server startup test failed: {e}")
        return False
    
    print("\n🎉 Startup test completed!")
    return True

if __name__ == "__main__":
    success = test_startup()
    sys.exit(0 if success else 1) 