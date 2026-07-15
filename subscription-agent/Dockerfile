FROM python:3.11-slim

WORKDIR /app

# System deps: none beyond what pip needs; keep the image small.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Unbuffered stdout so the Thought/Action/Observation trace streams live.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "app.main"]
