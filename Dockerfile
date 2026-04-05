FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CORE_ALLOWED_HOSTS="*"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY adapter ./adapter
COPY core ./core
COPY security ./security

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=3)"

CMD ["uvicorn", "adapter.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--proxy-headers"]
