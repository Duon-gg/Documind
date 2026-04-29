FROM python:3.11-slim

WORKDIR /app

# Build tools for C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install pydantic first to prevent conflicts
RUN pip install --no-cache-dir pydantic==2.9.2 pydantic-settings==2.8.1 pydantic_core==2.23.4

# Then install all deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data storage

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
