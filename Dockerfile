# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app
ENV HUGGINGFACE_HUB_TOKEN=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
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY app/ ./app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Note: Model download happens at runtime, not build time
# This allows using HuggingFace cache and makes builds faster

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
