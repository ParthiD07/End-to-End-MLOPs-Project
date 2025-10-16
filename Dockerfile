# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed to build some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy dependency file first (enables Docker caching)
COPY requirements.txt .

# Install awscli via pip (lightweight + fast)
RUN pip install --no-cache-dir awscli && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project code
COPY . .

# Ensure artifacts folder exists (if your app uses it)
RUN mkdir -p /app/artifacts

# Expose port if needed (e.g., Flask default 8080)
EXPOSE 8080

# Default command to run your app
CMD ["python", "app.py"]