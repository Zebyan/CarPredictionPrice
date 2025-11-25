FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install curl to download the model
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy backend source
COPY . /app/backend

# Ensure models_storage exists
RUN mkdir -p /app/backend/models_storage

# Download the model from GitHub Release
RUN curl -L "https://github.com/Zebyan/CarPredictionPrice/releases/download/rf-model/random_forest_light.pkl" \
    -o /app/backend/models_storage/random_forest_light.pkl

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
