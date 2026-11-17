FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY configs/ configs/

RUN pip install --no-cache-dir -e ".[all]"

# Run the pipeline on container start (synthetic data, no network needed).
CMD ["python", "-m", "alphalab.pipeline"]
