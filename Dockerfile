FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml setup.py requirements.txt *.cpp *.py /app/
COPY static /app/static

RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir -e .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "python -m uvicorn web_app:app --host 0.0.0.0 --port ${PORT}"]
