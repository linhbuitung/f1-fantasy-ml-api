import os
import base64
import subprocess

def dvc_pull_with_gcp_key():
    key_b64 = os.getenv('GCP_SA_KEY_B64')
    if key_b64:
        key_path = '/tmp/gcloud-key.json'
        with open(key_path, 'wb') as f:
            f.write(base64.b64decode(key_b64))
        subprocess.run(['dvc', 'config', 'core.no_scm', 'true'], check=True)
        subprocess.run(['dvc', 'remote', 'modify', '--local', 'myremote', 'credentialpath', key_path], check=True)
        subprocess.run(['dvc', 'pull', '-v'], check=True)
        os.remove(key_path)
    else:
        print('GCP_SA_KEY_B64 not provided; skipping dvc pull')
