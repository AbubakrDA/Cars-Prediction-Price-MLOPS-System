# 1. Use an explicit, robust python version
FROM python:3.11-slim

# 2. Set strict working directory
WORKDIR /app

# 3. Only copy required dependency file for the API serving layer to maximize caching
COPY requirements-api.txt .

# 4. Install dependencies efficiently
RUN pip install --no-cache-dir -r requirements-api.txt

# 5. Copy the rest of the application code
COPY src/ /app/src/

# We inject MLflow tracking URI securely at runtime.
# ENV MLFLOW_TRACKING_URI=...

# 6. Expose the port
EXPOSE 8000

# 7. Start the production server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
