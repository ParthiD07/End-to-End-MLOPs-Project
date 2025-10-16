# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (enables Docker caching)
COPY requirements.txt .

# Install Python dependencies and awscli
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends awscli && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Copy the rest of the project code
COPY . .

# Ensure artifacts folder exists (if your app uses it)
RUN mkdir -p /app/artifacts

# Expose port if needed (e.g., Flask default 8080)
EXPOSE 8080

# Default command to run your app
CMD ["python", "app.py"]