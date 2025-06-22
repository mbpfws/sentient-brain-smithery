#!/usr/bin/env python3
"""
Simple deployment validation script
Checks file structure and basic configuration without requiring dependencies
"""

import os
import json
import yaml
from pathlib import Path

def validate_deployment():
    """Validate deployment package structure and configuration"""
    print("🔍 Validating Sentient Brain Smithery Deployment Package")
    print("=" * 60)
    
    success = True
    
    # Check required files
    required_files = [
        "smithery.yaml",
        "smithery.json", 
        "Dockerfile",
        "mcp_server.py",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "DEPLOYMENT.md",
        "LICENSE",
        ".env.example",
        ".gitignore"
    ]
    
    print("1. Checking Required Files...")
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            success = False
    
    # Check directory structure
    print("\n2. Checking Directory Structure...")
    required_dirs = ["src"]
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ❌ {dir_name}/ - MISSING")
            success = False
    
    # Validate smithery.yaml
    print("\n3. Validating smithery.yaml...")
    try:
        with open("smithery.yaml", "r") as f:
            smithery_config = yaml.safe_load(f)
        
        required_keys = ["runtime", "build", "startCommand"]
        for key in required_keys:
            if key in smithery_config:
                print(f"   ✅ {key}")
            else:
                print(f"   ❌ {key} - MISSING")
                success = False
        
        # Check runtime type
        if smithery_config.get("runtime") == "container":
            print("   ✅ Runtime: container")
        else:
            print(f"   ❌ Runtime should be 'container', got: {smithery_config.get('runtime')}")
            success = False
            
    except Exception as e:
        print(f"   ❌ Error reading smithery.yaml: {e}")
        success = False
    
    # Validate smithery.json
    print("\n4. Validating smithery.json...")
    try:
        with open("smithery.json", "r") as f:
            smithery_meta = json.load(f)
        
        required_keys = ["id", "name", "description", "version", "tags"]
        for key in required_keys:
            if key in smithery_meta:
                print(f"   ✅ {key}: {smithery_meta[key] if key != 'tags' else f'{len(smithery_meta[key])} tags'}")
            else:
                print(f"   ❌ {key} - MISSING")
                success = False
                
    except Exception as e:
        print(f"   ❌ Error reading smithery.json: {e}")
        success = False
    
    # Check Dockerfile
    print("\n5. Validating Dockerfile...")
    try:
        with open("Dockerfile", "r") as f:
            dockerfile_content = f.read()
        
        required_patterns = [
            "FROM python:",
            "COPY requirements.txt",
            "RUN pip install",
            "COPY . .",
            "EXPOSE",
            "CMD"
        ]
        
        for pattern in required_patterns:
            if pattern in dockerfile_content:
                print(f"   ✅ {pattern}")
            else:
                print(f"   ❌ {pattern} - MISSING")
                success = False
                
    except Exception as e:
        print(f"   ❌ Error reading Dockerfile: {e}")
        success = False
    
    # Check Python files
    print("\n6. Validating Python Files...")
    python_files = ["mcp_server.py", "src/__init__.py"]
    for py_file in python_files:
        if Path(py_file).exists():
            try:
                with open(py_file, "r") as f:
                    content = f.read()
                if len(content) > 0:
                    print(f"   ✅ {py_file} ({len(content)} chars)")
                else:
                    print(f"   ❌ {py_file} - EMPTY")
                    success = False
            except Exception as e:
                print(f"   ❌ Error reading {py_file}: {e}")
                success = False
        else:
            print(f"   ❌ {py_file} - MISSING")
            success = False
    
    # Check requirements.txt
    print("\n7. Validating requirements.txt...")
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().strip().split('\n')
        
        required_packages = ["mcp", "fastapi", "uvicorn", "pydantic", "groq"]
        found_packages = []
        
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                package_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
                found_packages.append(package_name)
        
        for package in required_packages:
            if any(package in found for found in found_packages):
                print(f"   ✅ {package}")
            else:
                print(f"   ❌ {package} - MISSING")
                success = False
                
        print(f"   📦 Total packages: {len(found_packages)}")
        
    except Exception as e:
        print(f"   ❌ Error reading requirements.txt: {e}")
        success = False
    
    # Final validation
    print("\n" + "=" * 60)
    if success:
        print("🎉 DEPLOYMENT VALIDATION SUCCESSFUL!")
        print("✅ All required files and configurations are present")
        print("🚀 Ready for Smithery.ai deployment!")
        print("\nNext steps:")
        print("1. Push to GitHub repository")
        print("2. Connect repository to Smithery.ai")
        print("3. Configure environment variables")
        print("4. Deploy on Smithery.ai platform")
    else:
        print("❌ DEPLOYMENT VALIDATION FAILED!")
        print("🔧 Please fix the missing files/configurations above")
        print("📖 Refer to DEPLOYMENT.md for detailed instructions")
    
    return success

if __name__ == "__main__":
    try:
        success = validate_deployment()
        exit(0 if success else 1)
    except Exception as e:
        print(f"💥 Validation failed with error: {e}")
        exit(1)