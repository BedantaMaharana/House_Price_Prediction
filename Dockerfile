# ============================================================
# Dockerfile — House Price Prediction API
# ============================================================

# base image — slim keeps it lightweight
FROM python:3.12-slim

# set working directory inside container
WORKDIR /app

# copy requirements first — Docker caches this layer
# so if requirements don't change, it won't reinstall packages
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY app/ ./app/
COPY models/ ./models/

# expose the port FastAPI runs on
EXPOSE 8000

# command to run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
