FROM python:3.11-slim
ENV PORT=8080

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install uv

COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data
COPY ./scripts /app/scripts

# Sync dependencies (locked)
RUN uv sync --locked

COPY .dvc /app/.dvc
COPY dvc.lock /app/dvc.lock

CMD ["uv", "run", "app.main"]