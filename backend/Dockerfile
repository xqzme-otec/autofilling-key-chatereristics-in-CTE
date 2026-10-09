FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY aggregator ./aggregator

ENV MODEL_SERVICE_URL=http://model:8001
ENV REDIS_URL=redis://redis:6379/0
EXPOSE 8000
CMD ["uvicorn", "aggregator.main:app", "--host", "0.0.0.0", "--port", "8000"]
