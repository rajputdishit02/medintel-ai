"""Download an immutable NHANES source snapshot with provenance metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

import pandas as pd

from medintel.catalog import COMPONENTS, CYCLE, Component

USER_AGENT = "medintel-ai/0.2 (research data acquisition)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_xpt(path: Path, expected_component: str) -> dict[str, int]:
    """Confirm that a download is parseable and has a valid participant key."""
    frame = pd.read_sas(path, format="xport")
    if "SEQN" not in frame.columns:
        raise ValueError(f"{expected_component}: missing SEQN")
    if frame["SEQN"].isna().any():
        raise ValueError(f"{expected_component}: SEQN contains missing values")
    if frame["SEQN"].duplicated().any():
        raise ValueError(f"{expected_component}: SEQN contains duplicates")
    return {"rows": len(frame), "columns": len(frame.columns)}


def download_component(component: Component, raw_dir: Path) -> dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{component.code}.xpt"
    request = Request(component.data_url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response, NamedTemporaryFile(
        dir=raw_dir, suffix=".part", delete=False
    ) as temporary:
        content_type = response.headers.get_content_type()
        if content_type == "text/html":
            raise ValueError(f"{component.code}: server returned HTML, not XPT")
        while chunk := response.read(1024 * 1024):
            temporary.write(chunk)
        temporary_path = Path(temporary.name)
    try:
        validation = validate_xpt(temporary_path, component.code)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "component": component.code,
        "description": component.description,
        "source_url": component.data_url,
        "documentation_url": component.documentation_url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        **validation,
    }


def acquire_snapshot(data_dir: Path, *, overwrite: bool = False) -> Path:
    raw_dir = data_dir / "raw" / "nhanes-2021-2023"
    manifest_path = data_dir / "metadata" / "source-manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} already exists; use --overwrite for a fresh snapshot"
        )
    records = [download_component(component, raw_dir) for component in COMPONENTS]
    manifest = {
        "dataset": "CDC NHANES",
        "cycle": CYCLE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "components": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def verify_snapshot(data_dir: Path) -> int:
    """Recheck every local file against its manifest and structural contract."""
    manifest_path = data_dir / "metadata" / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_dir = data_dir / "raw" / "nhanes-2021-2023"
    for record in manifest["components"]:
        path = raw_dir / f"{record['component']}.xpt"
        if not path.exists():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"{record['component']}: byte count does not match manifest")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{record['component']}: SHA-256 does not match manifest")
        validation = validate_xpt(path, str(record["component"]))
        if validation["rows"] != record["rows"] or validation["columns"] != record["columns"]:
            raise ValueError(f"{record['component']}: shape does not match manifest")
    return len(manifest["components"])
