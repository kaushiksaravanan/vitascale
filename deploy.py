"""Deploy VitaScale to Hugging Face Spaces."""
import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required.")

api = HfApi(token=HF_TOKEN)
username = api.whoami()["name"]
REPO_ID = f"{username}/vitascale"
print(f"Deploying to: {REPO_ID}")

try:
    create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        token=HF_TOKEN,
        exist_ok=True,
    )
    print(f"Space created/exists: {REPO_ID}")
except Exception as e:
    print(f"Create repo: {e}")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

files_to_upload = [
    "models.py",
    "env.py",
    "graders.py",
    "load_traces.py",
    "app.py",
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt",
    "README.md",
]

for f in files_to_upload:
    local_path = os.path.join(PROJECT_DIR, f)
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f,
            repo_id=REPO_ID,
            repo_type="space",
            token=HF_TOKEN,
        )
        print(f"  Uploaded: {f}")
    else:
        print(f"  MISSING: {f}")

print(f"\nDone! Space URL: https://huggingface.co/spaces/{REPO_ID}")
