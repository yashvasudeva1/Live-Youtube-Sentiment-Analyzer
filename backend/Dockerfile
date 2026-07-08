FROM python:3.10-slim

WORKDIR /app

# Copy the backend requirements and code
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the mlflow database and models if they exist in the root context
COPY mlflow.db .
# Assuming mlruns or models directory might be needed
COPY models ./models

COPY backend ./backend

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
