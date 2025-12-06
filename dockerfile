FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-ansi

COPY . .

# Configure DVC to use local storage in Docker container
RUN dvc remote modify localstore url /dvc-storage || \
    dvc remote add localstore /dvc-storage || true

# Pull DVC data from local storage
RUN dvc pull || true

CMD ["streamlit", "run", "src/wine_quality/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
