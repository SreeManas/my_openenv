# ── Build stage ───────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port (HuggingFace Spaces requires 7860)
EXPOSE 7860

# Health-check (optional but nice for orchestration)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/tasks')" || exit 1

# Start the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
