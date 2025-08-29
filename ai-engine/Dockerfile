# AI Operations Engine Dockerfile
# AI Operations Engine Dockerfile - Python 3.12 for Ubuntu 24.04
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies optimized for Python 3.12
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libhdf5-dev \
    pkg-config \
    curl \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r aiengine && useradd -r -g aiengine aiengine

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ai_engine/ ./ai_engine/
COPY pyproject.toml .

# Install the package in development mode
RUN pip install -e .

# Create directories for models and logs
RUN mkdir -p /var/lib/novacron/ai-models /var/log/novacron-ai && \
    chown -R aiengine:aiengine /var/lib/novacron /var/log/novacron-ai

# Switch to non-root user
USER aiengine

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8093/health || exit 1

# Expose port
EXPOSE 8093

# Run the application
CMD ["python", "-m", "uvicorn", "ai_engine.api.main:app", "--host", "0.0.0.0", "--port", "8093"]