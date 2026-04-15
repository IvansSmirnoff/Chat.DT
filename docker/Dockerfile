# Constrained NL2Cypher for BIM
# Python 3.11 Application Dockerfile

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for ifcopenshell and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-all-dev \
    libocct-data-exchange-dev \
    libocct-foundation-dev \
    libocct-modeling-algorithms-dev \
    libocct-modeling-data-dev \
    libocct-ocaf-dev \
    libocct-visualization-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt-dev \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching.
# Server image installs base + dev deps only. LLM providers
# (google-generativeai, openai, torch, transformers, outlines, ...) live in
# requirements-llm.txt and are intended to be installed in a Colab runtime
# or ad-hoc on top of this image when running cloud experiments.
COPY requirements-base.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# To enable cloud LLM providers in this image:
#   pip install -r requirements-llm.txt
# To enable local LLM + llama.cpp:
#   CMAKE_ARGS="-DGGML_NATIVE=ON" CMAKE_BUILD_PARALLEL_LEVEL=16 \
#     pip install llama-cpp-python --no-deps --force-reinstall

# Copy application code
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY tests/ /app/tests/

# Create data directory
RUN mkdir -p /app/data

# Server image is ETL/bundle/CLI only — keep the container up so
# `docker compose exec app …` can drive it. Run a specific tool via exec,
# or override the command with `docker compose run --rm app <cmd>`.
CMD ["sleep", "infinity"]
