# Simplified Dockerfile for Smithery.ai deployment - DEBUGGING VERSION
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies directly (no venv for simplicity)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code (exclude problematic files)
COPY mcp_server.py ./
COPY smithery.yaml ./
COPY smithery.json ./
COPY src/ ./src/
COPY README.md ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/temp

# Health check for Smithery monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Simple start command - run as root for debugging
CMD ["python", "mcp_server.py"]