FROM python:3.12-slim
ENV PORT=8080

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data
COPY ./scripts /app/scripts

COPY .dvc /app/.dvc
COPY dvc.lock /app/dvc.lock

CMD ["python", "main.py"]