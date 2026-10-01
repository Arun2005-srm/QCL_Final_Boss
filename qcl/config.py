from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import torch
import yaml


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if key not in {"label_map", "color_map"} and isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError("Cyclic config defaults")
    seen.add(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = cfg.pop("defaults", None)
    cfg = merge(load_config(path.parent / parent, seen), cfg) if parent else cfg
    if len(cfg["dataset"]["classes"]) < 2:
        raise ValueError("Include background and foreground for binary segmentation")
    if len(set(cfg["dataset"]["classes"])) != len(cfg["dataset"]["classes"]):
        raise ValueError("Class names must be unique")
    for key in ("batch_size", "eval_batch_size", "accumulation"):
        if cfg["runtime"][key] < 1:
            raise ValueError(f"runtime.{key} must be positive")
    if cfg["model"]["quantum_chunk"] < 1 or cfg["model"]["grid"] < 1:
        raise ValueError("Quantum chunk and spatial grid must be positive")
    if cfg["training"]["epochs"] < 1 or cfg["training"]["max_updates_per_domain"] < 1:
        raise ValueError("Training budgets must be positive")
    if cfg["runtime"]["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative")
    if cfg["model"]["encoder"] not in {"sam_vit_b", "synthetic"}:
        raise ValueError("Supported encoder: sam_vit_b (synthetic is test-only)")
    if cfg["model"]["encoder"] == "sam_vit_b" and cfg["dataset"]["size"] != 1024:
        raise ValueError("SAM ViT-B requires size 1024")
    if cfg["dataset"]["ignore_index"] in range(len(cfg["dataset"]["classes"])):
        raise ValueError("ignore_index must be outside the class vocabulary")
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
