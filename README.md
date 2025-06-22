# Sentient Brain Multi-Agent System for Smithery.ai

Advanced AI Code Developer system leveraging multi-agent architecture with SurrealDB unified data layer. Designed for seamless deployment on [Smithery.ai](https://smithery.ai) platform.

## 🚀 Features

- **Ultra Orchestrator**: Master agent coordinating multi-agent workflows
- **Architect Agent**: Intelligent project design and planning
- **Code Analysis**: Deep code understanding and semantic indexing
- **Knowledge Graph**: Unified memory layer with semantic search
- **Debug & Refactor**: Intelligent code improvement and error resolution
- **Failure Prevention**: Advanced mechanisms to prevent common AI failures

## 🏗️ Architecture

### Multi-Agent Framework
- **Ultra Orchestrator**: Routes tasks and manages agent coordination
- **Architect Agent**: Handles project planning and design
- **Code Analysis Agent**: Provides deep code understanding
- **Knowledge Search Agent**: Semantic search across project knowledge
- **Debug & Refactor Agent**: Code improvement and error resolution

### Technology Stack
- **Runtime**: Python 3.11+ with FastAPI
- **Database**: SurrealDB for unified data layer
- **LLM**: Groq API for high-performance inference
- **Framework**: LangGraph for agent workflows
- **Protocol**: MCP (Model Context Protocol) compatible

## 📦 Smithery.ai Deployment

This package is optimized for deployment on Smithery.ai platform using the **Custom Deploy** method.

### Prerequisites
- Smithery.ai account
- GitHub repository
- Required API keys (Groq, optional Google)

### Deployment Steps

1. **Repository Setup**
   ```bash
   git clone <your-repo>
   cd sentient-brain-smithery
   ```

2. **Configuration**
   - Ensure `smithery.yaml` and `Dockerfile` are in root
   - Configure required environment variables in Smithery dashboard

3. **Deploy on Smithery**
   - Connect your GitHub repository to Smithery
   - Navigate to Deployments tab
   - Click "Deploy" to build and host your container

### Required Configuration

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `GROQ_API_KEY` | Groq API key for LLM inference | Yes | - |
| `SURREAL_URL` | SurrealDB connection URL | Yes | `ws://localhost:8000/rpc` |
| `SURREAL_USER` | SurrealDB username | Yes | `root` |
| `SURREAL_PASS` | SurrealDB password | Yes | `root` |
| `GOOGLE_API_KEY` | Google GenAI API key | No | - |

## 🛠️ Available Tools

### Core Tools

1. **sentient-brain/orchestrate**
   - Master coordination and workflow management
   - Analyzes user intent and routes to appropriate agents

2. **sentient-brain/architect**
   - Project design and architecture planning
   - Technology stack recommendations

3. **sentient-brain/analyze-code**
   - Deep code analysis and understanding
   - Structure, quality, and dependency analysis

4. **sentient-brain/search-knowledge**
   - Semantic search across project knowledge graph
   - Multi-modal search (code, docs, concepts)

5. **sentient-brain/debug-assist**
   - Intelligent debugging and code improvement
   - Error resolution and refactoring suggestions

## 🔧 Usage Examples

### Basic Orchestration
```json
{
  "tool": "sentient-brain/orchestrate",
  "arguments": {
    "query": "I want to build a REST API for user authentication",
    "context": {
      "project_type": "web_api",
      "tech_stack": ["python", "fastapi"]
    }
  }
}
```

### Project Architecture
```json
{
  "tool": "sentient-brain/architect",
  "arguments": {
    "project_type": "web_api",
    "requirements": "User authentication with JWT tokens",
    "tech_stack": ["python", "fastapi", "postgresql"]
  }
}
```

### Knowledge Search
```json
{
  "tool": "sentient-brain/search-knowledge",
  "arguments": {
    "query": "authentication middleware implementation",
    "node_type": "code_chunk",
    "limit": 10
  }
}
```

## 🔍 Health Check

The server provides a health check endpoint:
- `GET /` - Basic server status
- `GET /mcp` - MCP protocol info and available tools

## 🐛 Debugging

### Common Issues

1. **Connection Issues**
   - Verify SurrealDB connection parameters
   - Check network connectivity

2. **API Key Issues**
   - Ensure Groq API key is valid and has sufficient credits
   - Verify API key format and permissions

3. **Configuration Issues**
   - Check Smithery configuration parameters
   - Verify environment variable mapping

### Logs
The server uses structured logging with configurable levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARNING`: Warning messages
- `ERROR`: Error conditions

## 📈 Performance

- **Optimized Docker**: Multi-stage build for minimal image size
- **Async Operations**: Full async/await support for high concurrency
- **Caching**: Intelligent caching for frequently accessed data
- **Connection Pooling**: Efficient database connection management

## 🔒 Security

- **Non-root User**: Container runs as non-privileged user
- **Input Validation**: Comprehensive input validation using Pydantic
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Secure Defaults**: Security-first configuration defaults

## 📚 Documentation

- [Smithery.ai Documentation](https://smithery.ai/docs)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [SurrealDB Documentation](https://surrealdb.com/docs)
- [Groq API Documentation](https://console.groq.com/docs)

## 🤝 Support

- **Issues**: [GitHub Issues](https://github.com/sentient-brain/smithery-deployment/issues)
- **Documentation**: [docs.sentient-brain.ai](https://docs.sentient-brain.ai)
- **Email**: support@sentient-brain.ai

## 📄 License

MIT License - see LICENSE file for details.

---

**Ready for Smithery.ai deployment!** 🚀

---

## 🇻🇳 Tài liệu Tiếng Việt

### 🧠 Công cụ Cốt lõi

1. **sentient-brain/orchestrate**
   - Điều phối chính và quản lý luồng công việc
   - Phân tích ý định người dùng và chuyển tiếp đến các agent phù hợp

2. **sentient-brain/architect**
   - Thiết kế dự án và lập kế hoạch kiến trúc
   - Đề xuất lựa chọn công nghệ

3. **sentient-brain/analyze-code**
   - Phân tích mã nguồn sâu và hiểu mã
   - Phân tích cấu trúc, chất lượng và phụ thuộc

4. **sentient-brain/search-knowledge**
   - Tìm kiếm ngữ nghĩa trên đồ thị tri thức dự án
   - Tìm kiếm đa phương tiện (mã nguồn, tài liệu, khái niệm)

5. **sentient-brain/debug-assist**
   - Gỡ lỗi thông minh và nâng cao chất lượng mã
   - Gợi ý sửa lỗi và refactor

## 🔧 Ví dụ Sử Dụng

### Điều phối cơ bản
```json
{
  "tool": "sentient-brain/orchestrate",
  "arguments": {
    "query": "Tôi muốn xây dựng REST API cho xác thực người dùng",
    "context": {
      "project_type": "web_api",
      "tech_stack": ["python", "fastapi"]
    }
  }
}
```

### Thiết kế Kiến trúc Dự án
```json
{
  "tool": "sentient-brain/architect",
  "arguments": {
    "project_type": "web_api",
    "requirements": "Xác thực người dùng bằng JWT",
    "tech_stack": ["python", "fastapi", "postgresql"]
  }
}
```

### Tìm kiếm Tri thức
```json
{
  "tool": "sentient-brain/search-knowledge",
  "arguments": {
    "query": "triển khai middleware xác thực",
    "node_type": "code_chunk",
    "limit": 10
  }
}
```

## 🔍 Kiểm tra Sức khỏe

Máy chủ cung cấp các endpoint kiểm tra sức khỏe:
- `GET /` - Kiểm tra trạng thái cơ bản
- `GET /mcp` - Thông tin giao thức MCP và danh sách công cụ khả dụng

## 🐛 Gỡ Lỗi

### Các Vấn Đề Thường Gặp

1. **Kết nối đến CSDL**
   - Kiểm tra lại thông tin kết nối SurrealDB
   - Xác nhận kết nối mạng

2. **Vấn đề API Key**
   - Đảm bảo API key Groq hợp lệ và còn hạn mức sử dụng
   - Kiểm tra định dạng và quyền truy cập của API key

3. **Vấn đề Cấu hình**
   - Kiểm tra lại các tham số cấu hình Smithery
   - Đảm bảo ánh xạ biến môi trường chính xác

### Ghi log
Máy chủ sử dụng logging có cấu trúc với các mức độ cấu hình:
- `DEBUG`: Thông tin gỡ lỗi chi tiết
- `INFO`: Thông báo hoạt động chung
- `WARNING`: Cảnh báo
- `ERROR`: Lỗi nghiêm trọng

## 📈 Hiệu Năng

- **Docker tối ưu**: Multi-stage build giúp giảm kích thước image
- **Hoạt động bất đồng bộ**: Hỗ trợ async/await cho hiệu suất cao
- **Bộ nhớ đệm**: Cache thông minh cho dữ liệu truy cập thường xuyên
- **Kết nối pooling**: Quản lý kết nối CSDL hiệu quả

## 🔒 Bảo Mật

- **Chạy không quyền root**: Container chạy bằng user không đặc quyền
- **Kiểm tra đầu vào**: Sử dụng Pydantic để kiểm tra dữ liệu vào
- **Giới hạn tốc độ**: Tích hợp rate limiting cho các endpoint
- **Cấu hình an toàn mặc định**: Ưu tiên bảo mật từ đầu

## 📚 Tài liệu

- [Tài liệu Smithery.ai](https://smithery.ai/docs)
- [Đặc tả giao thức MCP](https://spec.modelcontextprotocol.io/)
- [Tài liệu SurrealDB](https://surrealdb.com/docs)
- [Tài liệu Groq API](https://console.groq.com/docs)

## 🤝 Hỗ trợ

- **Báo lỗi/Góp ý**: [GitHub Issues](https://github.com/sentient-brain/smithery-deployment/issues)
- **Tài liệu**: [docs.sentient-brain.ai](https://docs.sentient-brain.ai)
- **Email**: support@sentient-brain.ai

## 📄 Giấy phép

MIT License - xem file LICENSE để biết chi tiết.

---

**Sẵn sàng triển khai trên Smithery.ai!** 🚀