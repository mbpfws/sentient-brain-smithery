#!/usr/bin/env python3
"""
Smithery deployment debug script
"""
import json
import yaml
import sys
from pathlib import Path

def debug_smithery_config():
    """Debug Smithery configuration files"""
    print("🔍 Smithery Configuration Debug")
    print("=" * 40)
    
    # Check smithery.yaml
    try:
        with open("smithery.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
        print("✅ smithery.yaml loaded successfully")
        print(f"   Name: {yaml_config.get('name')}")
        print(f"   Runtime: {yaml_config.get('runtime')}")
        print(f"   Start command: {yaml_config.get('start', {}).get('command')}")
        print(f"   Health check: {yaml_config.get('healthCheck', {}).get('path')}")
    except Exception as e:
        print(f"❌ smithery.yaml error: {e}")
        return False
    
    # Check smithery.json
    try:
        with open("smithery.json", "r") as f:
            json_config = json.load(f)
        print("✅ smithery.json loaded successfully")
        print(f"   ID: {json_config.get('id')}")
        print(f"   Version: {json_config.get('version')}")
        print(f"   Deployment requirements: {len(json_config.get('deployment', {}).get('requirements', []))}")
    except Exception as e:
        print(f"❌ smithery.json error: {e}")
        return False
    
    # Check Dockerfile
    try:
        with open("Dockerfile", "r") as f:
            dockerfile = f.read()
        print("✅ Dockerfile found")
        if "CMD" in dockerfile:
            cmd_line = [line for line in dockerfile.split('\n') if line.strip().startswith('CMD')]
            if cmd_line:
                print(f"   Command: {cmd_line[0].strip()}")
        if "EXPOSE" in dockerfile:
            expose_line = [line for line in dockerfile.split('\n') if line.strip().startswith('EXPOSE')]
            if expose_line:
                print(f"   Exposed port: {expose_line[0].strip()}")
    except Exception as e:
        print(f"❌ Dockerfile error: {e}")
        return False
    
    # Check if all required files exist
    required_files = [
        "mcp_server.py",
        "requirements.txt", 
        "smithery.yaml",
        "smithery.json",
        "Dockerfile"
    ]
    
    print(f"\n📁 Required Files Check:")
    all_present = True
    for file in required_files:
        if Path(file).exists():
            size = Path(file).stat().st_size
            print(f"   ✅ {file} ({size} bytes)")
        else:
            print(f"   ❌ {file} MISSING")
            all_present = False
    
    # Validate configuration consistency
    print(f"\n🔧 Configuration Consistency:")
    
    # Check port consistency
    yaml_port = yaml_config.get('start', {}).get('port', 8000)
    health_port = yaml_config.get('healthCheck', {}).get('port', 8000)
    
    if yaml_port == health_port:
        print(f"   ✅ Port consistency: {yaml_port}")
    else:
        print(f"   ❌ Port mismatch: start={yaml_port}, health={health_port}")
        all_present = False
    
    # Check command consistency
    yaml_cmd = yaml_config.get('start', {}).get('command', [])
    if yaml_cmd == ["python", "mcp_server.py"]:
        print(f"   ✅ Start command correct")
    else:
        print(f"   ❌ Start command issue: {yaml_cmd}")
        all_present = False
    
    print(f"\n{'✅ All checks passed!' if all_present else '❌ Issues found!'}")
    return all_present

if __name__ == "__main__":
    success = debug_smithery_config()
    sys.exit(0 if success else 1) 