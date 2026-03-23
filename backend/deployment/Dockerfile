# AI Trainer API Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/assets/output /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config.yaml
ENV UPLOAD_DIR=/app/uploads
ENV OUTPUT_DIR=/app/assets/output
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/live || exit 1

# Run the API
CMD ["python", "api/start_api.py", "--host", "0.0.0.0", "--port", "8000"]
