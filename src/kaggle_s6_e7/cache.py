"""Content-addressed Parquet cache for fold-level feature frames."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

CACHE_SCHEMA_VERSION = 1
_CHUNK_SIZE = 8 * 1024 * 1024

log = logging.getLogger(__name__)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    stat = source.stat()
    file_size = stat.st_size
    digest = hashlib.sha256()
    pbar = tqdm(
        total=file_size,
        unit="B",
        unit_scale=True,
        desc=f"SHA256 {source.name}",
        disable=file_size < 50 * 1024 * 1024,  # only show for files >= 50 MB
    )
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            pbar.update(len(chunk))
    pbar.close()
    return {"path": str(source.resolve()), "size": file_size, "sha256": digest.hexdigest()}


class FoldFeatureCache:
    def __init__(self, root: str | Path = "outputs/cache", enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled

    def key(self, *, data: dict[str, Any], fold: int, config: dict[str, Any]) -> str:
        return stable_hash(
            {"schema": CACHE_SCHEMA_VERSION, "data": data, "fold": fold, "config": config}
        )

    def load(self, key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        directory = self.root / "folds" / key
        manifest_path = directory / "manifest.json"
        if not self.enabled or not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text())
            frames = tuple(pd.read_parquet(directory / f"{name}.parquet") for name in ("train", "valid", "test"))
            for name, frame in zip(("train", "valid", "test"), frames, strict=True):
                expected = manifest["frames"][name]
                if list(frame.columns) != expected["columns"] or len(frame) != expected["rows"]:
                    log.warning("Cache integrity check failed for %s/%s", key, name)
                    return None
            log.debug("Cache HIT: %s (%d train, %d valid, %d test rows)",
                      key, len(frames[0]), len(frames[1]), len(frames[2]))
            return frames  # type: ignore[return-value]
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def save(
        self, key: str, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
    ) -> None:
        if not self.enabled:
            return
        directory = self.root / "folds" / key
        directory.mkdir(parents=True, exist_ok=True)
        frames = {"train": train, "valid": valid, "test": test}
        for name, frame in frames.items():
            temp = directory / f".{name}.{os.getpid()}.parquet"
            frame.to_parquet(temp, index=False)
            temp.replace(directory / f"{name}.parquet")
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "frames": {
                name: {"rows": len(frame), "columns": list(frame.columns)}
                for name, frame in frames.items()
            },
        }
        temp_manifest = directory / f".manifest.{os.getpid()}.json"
        temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        temp_manifest.replace(directory / "manifest.json")


def load_cached_csv(path: str | Path, root: str | Path = "outputs/cache/raw") -> pd.DataFrame:
    """Load an immutable CSV through a content-addressed Parquet representation."""
    source = Path(path)
    fingerprint = file_fingerprint(source)
    cache_dir = Path(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    parquet = cache_dir / f"{source.stem}-{fingerprint['sha256'][:20]}.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    frame = pd.read_csv(source)
    temp = parquet.with_name(f".{parquet.name}.{os.getpid()}")
    frame.to_parquet(temp, index=False)
    temp.replace(parquet)
    return frame
