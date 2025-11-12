FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_lstm.py .

# Create directories for data and models
RUN mkdir -p /data /models /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import tensorflow as tf; print(tf.__version__)" || exit 1

# Default command
CMD ["python3", "train_lstm.py", \
     "--data", "/data/training_data.csv", \
     "--output", "/models", \
     "--epochs", "100", \
     "--batch-size", "32"]
