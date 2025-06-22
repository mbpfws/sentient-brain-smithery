# Platform Setup Guide for Sentient Brain MCP Server

This guide provides detailed setup instructions for connecting the Sentient Brain MCP Server to various AI platforms available through [Smithery.ai](https://smithery.ai).

## 🎯 Supported Platforms

Our MCP server supports **21 different AI platforms** through Smithery:

### **Desktop AI Clients**
- **Claude Desktop** - Anthropic's desktop client
- **Cursor** - AI-powered code editor
- **Windsurf** - AI development environment
- **VS Code** - With MCP extensions
- **VS Code Insiders** - Preview version
- **Cline** - AI coding assistant

### **Productivity & Workflow**
- **Raycast** - macOS productivity tool
- **Tome** - AI presentation tool
- **Highlight** - Research and note-taking
- **Asari** - AI assistant
- **Roo Code** - Code generation tool
- **Augment** - AI productivity suite
- **BoltAI** - AI chat interface
- **Goose** - AI development assistant
- **Witsy** - AI writing assistant
- **Enconvo** - AI workflow automation

### **Enterprise & Cloud**
- **Amazon Bedrock** - AWS AI services
- **Amazon Q** - AWS AI assistant

---

## 🚀 Quick Start

### 1. Deploy on Smithery
```bash
# Clone the repository
git clone https://your-repo/sentient-brain
cd sentient-brain/sentient-brain-smithery

# Deploy to Smithery
# Visit https://smithery.ai/deploy and connect your GitHub repo
```

### 2. Get Your Server URL
After deployment, you'll receive a URL like:
```
https://server.smithery.ai/@yourusername/sentient-brain
```

---

## 📱 Platform-Specific Setup

### Claude Desktop

**Configuration File Location:**
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Setup Steps:**
1. Open the configuration file
2. Add your server configuration:

```json
{
  "mcpServers": {
    "sentient-brain": {
      "command": "npx",
      "args": [
        "@smithery/cli",
        "run",
        "@yourusername/sentient-brain",
        "--config",
        "{\"GROQ_API_KEY\":\"your-groq-key\",\"SURREAL_URL\":\"ws://localhost:8000/rpc\"}"
      ]
    }
  }
}
```

3. Restart Claude Desktop

### Cursor IDE

**Setup Steps:**
1. Open Cursor settings (Cmd/Ctrl + ,)
2. Navigate to **Extensions** > **MCP Servers**
3. Add new server:

```json
{
  "name": "Sentient Brain",
  "url": "https://server.smithery.ai/@yourusername/sentient-brain",
  "config": {
    "GROQ_API_KEY": "your-groq-key",
    "SURREAL_URL": "ws://localhost:8000/rpc",
    "CURSOR_INTEGRATION": true
  }
}
```

4. Enable the server in Cursor's AI panel

### Windsurf

**Configuration File:** `.windsurf/mcp.json` in your project root

```json
{
  "mcpServers": {
    "sentient-brain": {
      "serverUrl": "https://server.smithery.ai/@yourusername/sentient-brain",
      "apiKey": "your-smithery-api-key",
      "config": {
        "GROQ_API_KEY": "your-groq-key",
        "SURREAL_URL": "ws://localhost:8000/rpc",
        "WINDSURF_INTEGRATION": true
      }
    }
  }
}
```

### VS Code

**Method 1: Using Smithery Extension**
1. Install the **Smithery MCP Client** extension
2. Open Command Palette (Cmd/Ctrl + Shift + P)
3. Run: `Smithery: Add MCP Server`
4. Enter server URL: `https://server.smithery.ai/@yourusername/sentient-brain`

**Method 2: Manual Configuration**
1. Create `.vscode/mcp-servers.json`:

```json
{
  "servers": {
    "sentient-brain": {
      "url": "https://server.smithery.ai/@yourusername/sentient-brain",
      "config": {
        "GROQ_API_KEY": "your-groq-key",
        "SURREAL_URL": "ws://localhost:8000/rpc"
      }
    }
  }
}
```

### Raycast (macOS)

1. Install **Raycast MCP Extension**
2. Open Raycast preferences
3. Navigate to **Extensions** > **MCP Servers**
4. Add server with URL: `https://server.smithery.ai/@yourusername/sentient-brain`

### Amazon Bedrock

**Using AWS CLI:**
```bash
# Configure MCP server in Bedrock
aws bedrock configure-mcp-server \
  --server-name "sentient-brain" \
  --server-url "https://server.smithery.ai/@yourusername/sentient-brain" \
  --config-json '{"GROQ_API_KEY":"your-key"}'
```

---

## ⚙️ Configuration Options

### Required Configuration
```yaml
GROQ_API_KEY: "gsk_your_groq_api_key"  # Required
```

### Database Configuration
```yaml
SURREAL_URL: "ws://localhost:8000/rpc"     # Local development
SURREAL_URL: "wss://your-db.fly.dev/rpc"  # Production
SURREAL_USER: "root"
SURREAL_PASS: "your_password"
SURREAL_NAMESPACE: "sentient_brain"
SURREAL_DATABASE: "multi_agent"
```

### Optional AI Services
```yaml
GOOGLE_API_KEY: "AIzaSy_your_google_key"      # For embeddings
OPENAI_API_KEY: "sk-your_openai_key"          # Additional models
ANTHROPIC_API_KEY: "sk-ant-your_anthropic_key" # Claude models
```

### Performance Tuning
```yaml
MAX_AGENTS: 5              # Concurrent agents (1-20)
AGENT_TIMEOUT: 300         # Timeout in seconds
MEMORY_LIMIT_MB: 1024      # Memory limit
VECTOR_DIMENSIONS: 1536    # Embedding dimensions
API_RATE_LIMIT: 100        # Requests per minute
```

### Platform-Specific Features
```yaml
CURSOR_INTEGRATION: true      # Enable Cursor-specific features
WINDSURF_INTEGRATION: true    # Enable Windsurf-specific features
CLAUDE_DESKTOP_INTEGRATION: true  # Enable Claude Desktop features
```

---

## 🔧 Local Development Setup

### Prerequisites
- Python 3.11+
- SurrealDB running locally
- Groq API key

### Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Start SurrealDB
surreal start --log trace --user root --pass root memory

# Run MCP server locally
python mcp_server.py

# Test with MCP Inspector
npx @modelcontextprotocol/inspector http://localhost:8000/mcp
```

### Local MCP Configuration
For local development, use `stdio` transport:

```json
{
  "mcpServers": {
    "sentient-brain-local": {
      "command": "python",
      "args": ["path/to/sentient-brain/mcp_server.py"],
      "env": {
        "GROQ_API_KEY": "your-key",
        "SURREAL_URL": "ws://localhost:8000/rpc"
      }
    }
  }
}
```

---

## 🔍 Testing & Troubleshooting

### Health Check
Visit your server's health endpoint:
```
https://server.smithery.ai/@yourusername/sentient-brain/health
```

### Common Issues

**1. Tool Discovery Fails**
- Ensure `GROQ_API_KEY` is set correctly
- Check server logs for authentication errors
- Verify network connectivity

**2. SurrealDB Connection Issues**
- Confirm SurrealDB is running and accessible
- Check WebSocket URL format (`ws://` or `wss://`)
- Verify credentials and permissions

**3. Platform-Specific Issues**
- **Claude Desktop:** Restart the application after config changes
- **Cursor:** Check the AI panel for connection status
- **VS Code:** Reload window after installing extensions

### Debug Mode
Enable debug logging:
```yaml
DEBUG_MODE: true
LOG_LEVEL: "DEBUG"
```

### Monitoring
Enable metrics collection:
```yaml
ENABLE_METRICS: true
ENABLE_TRACING: true
```

---

## 📚 Additional Resources

- **Smithery Documentation:** https://smithery.ai/docs
- **MCP Specification:** https://modelcontextprotocol.io/
- **SurrealDB Docs:** https://surrealdb.com/docs
- **Groq API Docs:** https://console.groq.com/docs

## 🆘 Support

- **GitHub Issues:** [Report bugs and feature requests](https://github.com/your-username/sentient-brain/issues)
- **Smithery Discord:** [Community support](https://discord.gg/smithery)
- **Documentation:** [Detailed guides](https://github.com/your-username/sentient-brain/wiki)

---

## 🎉 Ready to Deploy?

1. **Star the repository** ⭐
2. **Deploy to Smithery** 🚀
3. **Configure your favorite AI platform** 🤖
4. **Start building with multi-agent AI** 🧠

Happy coding! 🎯 