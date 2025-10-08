FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data
COPY ./scripts /app/scripts

COPY .dvc /app/.dvc
COPY dvc.yaml /app/dvc.yaml
COPY dvc.lock /app/dvc.lock