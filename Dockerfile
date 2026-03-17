FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY web ./web
COPY artifacts ./artifacts
COPY reports ./reports

EXPOSE 8000

CMD ["uvicorn", "malware_detector.api.app:app", "--host", "0.0.0.0", "--port", "8000"]