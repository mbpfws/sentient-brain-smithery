# Simplified Dockerfile for Smithery.ai deployment
FROM python:3.11-slim

# Set environment variables to sane defaults
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY mcp_server.py ./
COPY src/ ./src/

# Create data directory for the SQLite database
RUN mkdir -p /app/data

# Expose the port the app runs on
EXPOSE ${PORT}

# The command to run the application
# Smithery uses the healthCheck in smithery.yaml, so a Docker HEALTHCHECK is not needed.
CMD ["python", "mcp_server.py"]