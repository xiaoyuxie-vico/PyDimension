# ---- Build stage: install dependencies in a clean layer ----
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the build-time system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first (saves ~1.5 GB vs full CUDA build),
# then the rest of the requirements.
RUN pip install --no-cache-dir --prefix=/install \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install \
        -r requirements.txt

# ---- Runtime stage ----
FROM python:3.11-slim

LABEL maintainer="PyDimension Team"
LABEL description="PyDimension: data-driven discovery of dimensionless numbers and symmetries"

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy project source
COPY . .

# Install the package itself (editable so scripts at repo root still work)
RUN pip install --no-cache-dir -e .

# Streamlit config: disable telemetry, bind to 0.0.0.0
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

EXPOSE 8501

# Default: launch the Streamlit web app.
# Override with e.g.  docker run pydimension python run_pipeline.py --help
CMD ["streamlit", "run", "streamlit_app.py"]
