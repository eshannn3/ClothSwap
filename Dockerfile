FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the TensorFlow model file exists
# If the model file is missing, add it to your project directory
COPY Model2.h5 /app/Model2.h5

# Set environment variable to suppress TensorFlow warnings
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app
