"""Deploy updated VitaScale files."""
import os
from huggingface_hub import HfApi

HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO = os.environ.get("HF_REPO_ID", "kaushikss/vitascale")
BASE = os.path.dirname(os.path.abspath(__file__))

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable is required.")

api = HfApi(token=HF_TOKEN)

for f in ["README.md", "inference.py"]:
    api.upload_file(
        path_or_fileobj=os.path.join(BASE, f),
        path_in_repo=f,
        repo_id=REPO,
        repo_type="space",
    )
    print(f"Uploaded: {f}")
print("Done!")
