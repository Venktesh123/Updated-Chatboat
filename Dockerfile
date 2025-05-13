# Use Python 3.9 as the base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create the directory for vectordb if it doesn't exist
RUN mkdir -p vectordb

# Set environment variables
ENV FLASK_APP=app5.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8000

# Expose the port the app runs on
EXPOSE 8000

# Initialize the vector store (only if cleaned_transcript.txt exists)
RUN if [ -f cleaned_transcript.txt ]; then python vectordb.py; fi

# Use Gunicorn to run the application with Gevent workers
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "gevent", "--timeout", "120", "app5:app"]