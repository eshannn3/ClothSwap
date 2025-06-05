FROM python:3.9-slim

WORKDIR /app

# Install git and git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    gcc \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure git-lfs files are pulled (if needed)
RUN git lfs pull || echo "LFS files already present"

# Set environment variable to suppress TensorFlow warnings
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app
