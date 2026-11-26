FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src TECHCART_DATA_DIR=/data
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY scripts ./scripts
# non-root user
RUN useradd -m app && mkdir -p /data && chown -R app /data
USER app
EXPOSE 8000
# first start: build data, models and index if the volume is empty, then serve
CMD ["sh", "-c", "[ -f /data/techcart.db ] || (python scripts/seed.py && python scripts/train_all.py && python scripts/build_index.py); uvicorn techcart.api.main:app --host 0.0.0.0 --port 8000"]
