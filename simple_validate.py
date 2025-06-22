#!/usr/bin/env python3
"""Simple validation without external dependencies"""

import os
import json
from pathlib import Path

def main():
    print("🔍 Validating Smithery Deployment Package")
    print("=" * 45)
    
    # Check essential files
    files = [
        "smithery.yaml", "smithery.json", "Dockerfile", 
        "mcp_server.py", "requirements.txt", "README.md"
    ]
    
    all_good = True
    for file in files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} MISSING")
            all_good = False
    
    # Check smithery.json structure
    try:
        with open("smithery.json") as f:
            config = json.load(f)
        print(f"✅ smithery.json valid - ID: {config.get('id', 'unknown')}")
    except Exception as e:
        print(f"❌ smithery.json error: {e}")
        all_good = False
    
    if all_good:
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("🚀 Ready for Smithery.ai deployment!")
        return True
    else:
        print("\n❌ VALIDATION FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)