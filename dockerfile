FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3
RUN pip install "poetry==$POETRY_VERSION"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-ansi

COPY . .

CMD ["streamlit", "run", "src/wine_quality/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
