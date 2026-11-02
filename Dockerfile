FROM python:3.11-slim

# Install system dependencies for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage build caching
COPY requirements.txt .

# Install standard Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set default port expected by Cloud Run
ENV PORT=8080

# Run the app
CMD ["python", "app.py"]
