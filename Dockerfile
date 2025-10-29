FROM python:3.11-slim
ENV PORT=8080

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock /app/
COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data
COPY ./scripts /app/scripts
COPY .dvc /app/.dvc
COPY dvc.lock /app/dvc.lock

# Sync dependencies (locked)
RUN uv sync --no-dev

CMD ["uv", "run", "python", "-m", "app.main"]