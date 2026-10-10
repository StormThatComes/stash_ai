"""Pinned upstream JoyTag architecture, downloaded from the author's HF Space."""
import hashlib
import importlib.util
import json
from pathlib import Path

LOADER_REPO = "fancyfeast/joytag"
LOADER_REVISION = "bce35c9afc4dbe5eb71ee836939c84d4614eec96"
LOADER_SHA256 = "48a39ee5e643ae06f64d0001305d7c8333c22ef22c139b9b619766c5ff8ce290"


def loader_path(download_missing=True):
    """Download/cache source without importing torch or executing model code."""
    from huggingface_hub import hf_hub_download
    path = Path(hf_hub_download(repo_id=LOADER_REPO, repo_type="space",
        filename="Models.py", revision=LOADER_REVISION,
        local_files_only=not download_missing))
    if hashlib.sha256(path.read_bytes()).hexdigest() != LOADER_SHA256:
        raise ValueError("JoyTag loader checksum mismatch; remove the damaged HF cache entry and download again.")
    return path


def load_model(model_dir, label_count, download_missing=True):
    """Use the upstream architecture with safetensors only, without sys.path edits."""
    from safetensors.torch import load_file
    path = loader_path(download_missing)
    # Compile the verified bytes directly, avoiding stale Python bytecode and
    # any local module named Models. The upstream module has no relative imports.
    spec = importlib.util.spec_from_file_location("_stash_ai_joytag", path)
    module = importlib.util.module_from_spec(spec)
    # The executed source, not just an earlier file read, must match the pin.
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != LOADER_SHA256:
        raise ValueError("JoyTag loader changed after verification; refusing execution.")
    exec(compile(source, str(path), "exec"), module.__dict__)
    config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("class") != "ViT" or config.get("n_tags") != label_count:
        raise ValueError("JoyTag config must describe the Hugging Face ViT and match top_tags.txt.")
    model = module.ViT(**{key: value for key, value in config.items() if key != "class"})
    model.load(load_file(str(Path(model_dir) / "model.safetensors"), device="cpu"))
    return model
