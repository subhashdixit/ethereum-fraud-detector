"""Fetch the original Kaggle dataset without committing raw records."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def download():
    """Fetch one configured CSV and compare it with the reviewed training hash."""
    config = json.loads((ROOT / "configs/project.json").read_text())
    target = ROOT / "data" / config["file"]
    target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "dataset.zip"
        with urlopen(config["download"], timeout=60) as response:
            with archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as dataset:
            matches = [
                name for name in dataset.namelist() if Path(name).name == config["file"]
            ]
            if len(matches) != 1:
                raise ValueError("Expected exactly one configured source CSV")
            if dataset.getinfo(matches[0]).file_size > 300_000_000:
                raise ValueError("Source file exceeds expected size limit")
            content = dataset.read(matches[0])
        digest = hashlib.sha256(content).hexdigest()
        report = ROOT / "models/metrics.json"
        if (
            report.exists()
            and json.loads(report.read_text())["source_sha256"] != digest
        ):
            raise ValueError("Source dataset changed; review before retraining")
        target.write_bytes(content)
    print("Downloaded", target.name, "SHA-256:", digest)


if __name__ == "__main__":
    download()
