FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY models ./models
COPY artifacts ./artifacts
COPY README.md ./README.md

RUN mkdir -p logs artifacts

CMD ["python", "src/inference.py", "--input", "data/sample_input.csv", "--output", "artifacts/docker_predictions.csv"]

