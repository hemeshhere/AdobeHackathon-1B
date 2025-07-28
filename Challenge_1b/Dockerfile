# Use AMD64 Python 3.10 slim
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY run_analysis.py .
COPY schema/ ./schema

# Entry point
ENTRYPOINT ["python", "run_analysis.py"]
