FROM python:3.11-slim

# Prevent .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Project root inside container
WORKDIR /app

# Install dependencies (requirements.txt is in your backend folder)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend package into /app/backend
# So inside the container you'll have:
# /app/backend/main.py
# /app/backend/config.py
# /app/backend/models, routes, services, ...
COPY . /app/backend

# Make sure Python can see the "backend" package
ENV PYTHONPATH=/app

EXPOSE 8000

# Run FastAPI app: backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
