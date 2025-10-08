#!/usr/bin/env bash
set -euo pipefail

# Expect GCP_SA_KEY_B64 in env (base64 encoded JSON)
if [ -n "${GCP_SA_KEY_B64:-}" ]; then
  echo "$GCP_SA_KEY_B64" | base64 --decode > /tmp/gcloud-key.json
  export GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcloud-key.json"
  # make dvc use this credential file for the remote (local override)
  dvc remote modify --local myremote credentialpath /tmp/gcloud-key.json
fi

# ensure dvc is installed in runtime (see [requirements.txt](requirements.txt))
dvc pull -q