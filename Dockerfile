FROM python:3.12-slim

ARG GCP_SA_KEY_B64
ENV GCP_SA_KEY_B64=${GCP_SA_KEY_B64}

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

# Decode base64-encoded key and write file (safer than raw JSON in ARG)
RUN if [ -n "$GCP_SA_KEY_B64" ]; then \
      echo "$GCP_SA_KEY_B64" | base64 -d > /tmp/gcloud-key.json ; \
      dvc config core.no_scm true ; \
      dvc remote modify --local myremote credentialpath /tmp/gcloud-key.json ; \
      dvc pull -v || { echo "DVC pull failed"; exit 1; } ; \
      rm -f /tmp/gcloud-key.json ; \
    else \
      echo "GCP_SA_KEY_B64 not provided; skipping dvc pull at build time"; \
    fi

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

# log that the url is http://localhost:8000/docs
RUN echo "Application will be available at http://localhost:8080/docs"