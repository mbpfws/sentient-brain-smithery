# Smithery.ai Deployment Guide

Complete guide for deploying Sentient Brain Multi-Agent System on Smithery.ai platform.

## 🎯 Overview

This deployment package is specifically optimized for [Smithery.ai](https://smithery.ai) using the **Custom Deploy** method with Docker containers.

## 📋 Prerequisites

### Required Accounts & Keys
- [Smithery.ai](https://smithery.ai) account
- [GitHub](https://github.com) repository access
- [Groq](https://console.groq.com) API key (required)
- [Google AI](https://ai.google.dev) API key (optional)

### System Requirements
- Docker support (handled by Smithery)
- HTTP endpoint capability
- Environment variable configuration

## 🚀 Deployment Process

### Step 1: Repository Setup

1. **Fork/Clone Repository**
   ```bash
   git clone https://github.com/your-username/sentient-brain-smithery.git
   cd sentient-brain-smithery
   ```

2. **Verify Files**
   Ensure these files are present:
   - `smithery.yaml` - Smithery configuration
   - `Dockerfile` - Container definition
   - `mcp_server.py` - Main server application
   - `requirements.txt` - Python dependencies

### Step 2: Smithery Configuration

1. **Connect GitHub**
   - Login to [Smithery.ai](https://smithery.ai)
   - Connect your GitHub account
   - Select your repository

2. **Configure Environment**
   Set these required variables in Smithery:
   
   | Variable | Value | Description |
   |----------|-------|-------------|
   | `GROQ_API_KEY` | `gsk_your_key_here` | Groq API key |
   | `SURREAL_URL` | `ws://localhost:8000/rpc` | SurrealDB URL |
   | `SURREAL_USER` | `root` | Database username |
   | `SURREAL_PASS` | `your_password` | Database password |

3. **Optional Configuration**
   | Variable | Default | Description |
   |----------|---------|-------------|
   | `GOOGLE_API_KEY` | - | Google AI API key |
   | `LOG_LEVEL` | `INFO` | Logging level |
   | `GROQ_MODEL` | `llama-3.1-70b-versatile` | Groq model |

### Step 3: Deploy

1. **Navigate to Deployments**
   - Go to your server page on Smithery
   - Click "Deployments" tab

2. **Deploy Container**
   - Click "Deploy" button
   - Smithery will build your Docker container
   - Wait for deployment to complete

3. **Verify Deployment**
   - Check deployment status
   - Test health endpoint
   - Verify MCP protocol response

## 🔧 Configuration Details

### Smithery.yaml Breakdown

```yaml
runtime: "container"              # Use Docker container
build:
  dockerfile: "Dockerfile"        # Dockerfile location
  dockerBuildPath: "."           # Build context
startCommand:
  type: "http"                   # HTTP endpoint type
  configSchema:                  # Configuration schema
    type: object
    required: ["GROQ_API_KEY", "SURREAL_URL", "SURREAL_USER", "SURREAL_PASS"]
    properties:
      GROQ_API_KEY:
        type: string
        description: "Groq API key for LLM inference"
      # ... other properties
```

### Dockerfile Optimization

The Dockerfile uses multi-stage builds:
- **Builder stage**: Installs dependencies
- **Production stage**: Minimal runtime image
- **Security**: Non-root user execution
- **Health checks**: Built-in health monitoring

## 🌐 MCP Protocol Endpoints

### HTTP Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/` | Health check |
| `GET` | `/mcp` | MCP server info |
| `POST` | `/mcp` | Tool execution |
| `DELETE` | `/mcp` | Cleanup |

### Configuration Handling

Smithery passes configuration via query parameters:
```
GET /mcp?server.apiKey=sk-123&server.host=localhost
```

The server parses these into environment variables:
- `server.apiKey` → `SERVER_API_KEY`
- `server.host` → `SERVER_HOST`

## 🛠️ Tool Discovery

### Available Tools

1. **sentient-brain/orchestrate**
   - Master workflow coordination
   - Intent analysis and routing

2. **sentient-brain/architect**
   - Project design and planning
   - Architecture recommendations

3. **sentient-brain/analyze-code**
   - Code analysis and understanding
   - Quality and structure assessment

4. **sentient-brain/search-knowledge**
   - Knowledge graph semantic search
   - Multi-modal content retrieval

5. **sentient-brain/debug-assist**
   - Intelligent debugging
   - Code improvement suggestions

### Tool Usage Pattern

```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": {
    "name": "sentient-brain/orchestrate",
    "arguments": {
      "query": "Build a REST API",
      "context": {"type": "web_api"}
    }
  }
}
```

## 🔍 Testing & Validation

### Health Checks

1. **Basic Health**
   ```bash
   curl https://your-deployment.smithery.ai/
   ```

2. **MCP Protocol**
   ```bash
   curl https://your-deployment.smithery.ai/mcp
   ```

3. **Tool Execution**
   ```bash
   curl -X POST https://your-deployment.smithery.ai/mcp \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","method":"tools/list","id":"1"}'
   ```

### Validation Checklist

- [ ] Container builds successfully
- [ ] Server starts without errors
- [ ] Health endpoint responds
- [ ] MCP protocol endpoints work
- [ ] Tools are discoverable
- [ ] Configuration is loaded correctly
- [ ] Logging is functioning

## 🐛 Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Dockerfile syntax
   - Verify requirements.txt dependencies
   - Ensure Python version compatibility

2. **Runtime Errors**
   - Verify API keys are valid
   - Check database connectivity
   - Review environment variables

3. **Configuration Issues**
   - Validate smithery.yaml syntax
   - Check required vs optional parameters
   - Verify query parameter parsing

### Debug Commands

```bash
# Check container logs
docker logs <container-id>

# Test locally
docker build -t sentient-brain .
docker run -p 8000:8000 -e GROQ_API_KEY=your_key sentient-brain

# Validate configuration
curl localhost:8000/mcp
```

## 📊 Performance Optimization

### Container Optimization
- Multi-stage builds reduce image size
- Non-root user improves security
- Health checks enable monitoring

### Application Optimization
- Async/await for concurrency
- Connection pooling for databases
- Caching for frequently accessed data

### Resource Management
- Memory limits in Docker
- CPU optimization
- Disk space management

## 🔒 Security Considerations

### Container Security
- Non-root user execution
- Minimal base image
- Security-focused defaults

### API Security
- Input validation with Pydantic
- Rate limiting capabilities
- Secure environment handling

### Data Security
- Encrypted API keys
- Secure database connections
- Audit logging

## 📈 Monitoring & Maintenance

### Health Monitoring
- Built-in health checks
- Structured logging
- Error tracking

### Performance Monitoring
- Response time tracking
- Resource usage monitoring
- Error rate analysis

### Maintenance Tasks
- Regular dependency updates
- Security patch management
- Performance optimization

## 🆘 Support Resources

### Documentation
- [Smithery.ai Docs](https://smithery.ai/docs)
- [MCP Protocol](https://spec.modelcontextprotocol.io/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Community Support
- [Smithery Discord](https://discord.gg/smithery)
- [GitHub Issues](https://github.com/sentient-brain/smithery-deployment/issues)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/smithery)

### Professional Support
- Email: support@sentient-brain.ai
- Documentation: docs.sentient-brain.ai
- Enterprise: enterprise@sentient-brain.ai

---

**Happy Deploying!** 🚀