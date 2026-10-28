FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN mkdir -p /app/models

ARG MODEL_FILE=Baseline.onnx
ARG MODEL_VERSION=baseline-fp32-v1

COPY models/${MODEL_FILE} /app/models/${MODEL_FILE}

ENV MODEL_PATH=/app/models/${MODEL_FILE}
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]