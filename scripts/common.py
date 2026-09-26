"""Shared helpers: config loading, paths, result I/O. Imported by every script."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(model_name: str) -> dict:
    """model_name is the yaml filename stem under configs/models/, e.g. 'qwen2.5-0.5b'."""
    path = CONFIGS_DIR / "models" / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config at {path}")
    return load_yaml(path)


def load_quant_config(config_id: str) -> dict:
    path = CONFIGS_DIR / "quant" / f"{config_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No quant config at {path}")
    return load_yaml(path)


def load_matrix() -> dict:
    return load_yaml(CONFIGS_DIR / "matrix.yaml")


def load_profile(profile_name: str) -> dict:
    path = CONFIGS_DIR / "profiles" / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No profile config at {path}")
    return load_yaml(path)


def all_config_ids() -> list[str]:
    return [c["id"] for c in load_matrix()["configs"]]


def isolation_ref_ids() -> set[str]:
    return {c["id"] for c in load_matrix()["configs"] if c.get("is_isolation_ref")}


@dataclass
class RunPaths:
    model_name: str
    config_id: str

    @property
    def results_dir(self) -> Path:
        d = RESULTS_DIR / self.model_name / self.config_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def checkpoint_dir(self) -> Path:
        # gitignored — quantized weights never go to GitHub, only JSON results do.
        d = REPO_ROOT / "checkpoints" / self.model_name / self.config_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def done_marker(self) -> Path:
        return self.results_dir / "DONE"

    @property
    def meta_path(self) -> Path:
        return self.results_dir / "meta.json"

    @property
    def eval_path(self) -> Path:
        return self.results_dir / "eval.json"

    def bench_path(self, workload: str, concurrency: int) -> Path:
        return self.results_dir / f"bench_{workload}_c{concurrency}.json"

    def is_done(self) -> bool:
        return self.done_marker.exists()

    def mark_done(self) -> None:
        self.done_marker.write_text("")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def config_meta(model_name: str, config_id: str) -> dict:
    """Bit-width / tier metadata used to annotate every result file, so collect.py
    never has to re-derive tier semantics from scratch."""
    quant = load_quant_config(config_id)
    matrix_entry = next(c for c in load_matrix()["configs"] if c["id"] == config_id)

    def bits_of(block: Optional[dict]) -> Optional[int]:
        if not block:
            return None
        scheme = block.get("scheme", "")
        if "4" in scheme:
            return 4
        if "8" in scheme:
            return 8
        return None

    return {
        "model_name": model_name,
        "config_id": config_id,
        "target": quant.get("target"),
        "tier": quant.get("tier"),
        "tier_axis": quant.get("tier_axis"),
        "is_isolation_ref": matrix_entry.get("is_isolation_ref", False),
        "w_bits": bits_of(quant.get("weights")),
        "a_bits": bits_of(quant.get("activations")),
        "kv_bits": quant["kv_cache"].get("num_bits") if quant.get("kv_cache", {}).get("enabled") else None,
    }
