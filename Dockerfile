FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="DarkStore Pricing Environment"
LABEL org.opencontainers.image.description="OpenEnv-compatible perishable goods pricing simulation"
LABEL org.opencontainers.image.version="1.0.0"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860
ENV ENABLE_WEB_INTERFACE=true

# Working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY darkstore_env.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run server
CMD ["python", "server.py"]
