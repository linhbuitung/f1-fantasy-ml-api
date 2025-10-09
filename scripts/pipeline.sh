#!/bin/bash

## Write GCP credentials to file
#echo "$GCP_SA_KEY" > gcloud-key.json
#
## Tell DVC to skip Git
#dvc config core.no_scm true
#
## Configure DVC to use the credentials
#dvc remote modify --local myremote credentialpath gcloud-key.json
#
## Pull data from remote
#dvc pull -v || { echo "DVC pull failed"; exit 1; }
#
## Optional cleanup
#rm gcloud-key.json

python scripts/rebuild_processed_data.py
python scripts/create_mainrace_training.py
python scripts/create_qualifying_training.py
python scripts/create_status_training.py
python scripts/train_mainrace_model.py
python scripts/train_qualifying_model.py
python scripts/train_status_model.py