FROM python:3.10-slim

WORKDIR /app

# Install system dependencies & supervisord
RUN apt-get update && apt-get install -y supervisor && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Patch httpx to 0.27.0 for Groq compatibility
RUN pip install --no-cache-dir httpx==0.27.0

# Copy application files
COPY . .

# Setup supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000 8501

CMD ["/usr/bin/supervisord"]
