FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    POETRY_REQUESTS_TIMEOUT=120

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libopenblas-dev \
        liblapack-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock README.md LICENSE CITATION.cff ./
RUN python -m pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-root \
    && pip install numba

COPY src ./src
COPY entry_points ./entry_points
COPY tests ./tests
COPY conf ./conf
COPY docs ./docs
COPY experiments ./experiments
COPY paper ./paper
COPY tools ./tools
COPY .github ./.github
COPY UNITS_CONTRACT.md ./
COPY data ./data

RUN pip install -e .

CMD ["python", "-m", "pytest", "-q"]
