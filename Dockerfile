FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only PyTorch keeps the image small
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

RUN useradd --create-home appuser
USER appuser

# Mount a trained run directory at /models/run (see README)
ENV CONFUSION_BERT_RUN_DIR=/models/run
EXPOSE 8000
CMD ["uvicorn", "confusion_bert.serving.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
