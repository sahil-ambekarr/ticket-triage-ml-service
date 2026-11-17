# Not verified locally (no Docker on the build machine).
FROM python:3.14-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data
COPY reports ./reports

# The model file is not committed; build it into the image from the committed CSV (about 1-2 minutes).
RUN python -m scripts.train

EXPOSE 4103
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4103"]
