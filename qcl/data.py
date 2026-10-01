"""Explicit labels and group-isolated 70/15/15 splits for RGB segmentation."""
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .config import file_hash, save_json


@dataclass(frozen=True)
class Record:
    id: str
    image: str
    mask: str
    group: str
    domain: str
    split: str = ""


def discover(cfg):
    c = cfg["dataset"]
    records = []
    if c["manifest"]:
        path = Path(c["manifest"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Manifest not found: {path}. Set dataset.manifest to the existing CSV; paths resolve from the working directory.")
        columns = {k: k for k in ("id", "image", "mask", "group", "domain", "split")}
        columns.update(c.get("columns", {}))
        split_source = c.get("split_source", "generated")
        if split_source not in {"generated", "manifest"}:
            raise ValueError("dataset.split_source must be generated or manifest")
        with path.open(newline="", encoding="utf-8-sig") as f:
            for number, row in enumerate(csv.DictReader(f), 2):
                values = {k: (row.get(col) or "").strip() for k, col in columns.items()}
                missing = [columns[k] for k in ("id", "image", "mask", "group") if not values[k]]
                if missing:
                    raise ValueError(f"{path}:{number}: missing nonempty columns {missing}; configure dataset.columns")
                domain = values["domain"]
                # Prepared OpenEarthMap manifests may omit a domain column and
                # encode the region in group as "region/source". Use that
                # prefix when the domain is absent; the config flag also lets
                # callers require/validate this convention when a domain is
                # explicitly supplied.
                if c.get("domain_from_group_prefix", False) or (not domain and "/" in values["group"]):
                    parts = values["group"].split("/", 1)
                    if len(parts) != 2 or not all(parts):
                        raise ValueError(f"{path}:{number}: expected region/source in group, got {values['group']!r}")
                    if domain and domain != parts[0]:
                        raise ValueError(f"{path}:{number}: domain conflicts with group region prefix")
                    domain = parts[0]
                domain = domain or c.get("default_domain", "")
                if not domain:
                    raise ValueError(f"{path}:{number}: supply domain column, default_domain, or domain_from_group_prefix")
                assigned = values["split"]
                if split_source == "manifest" and assigned not in {"train", "val", "test"}:
                    raise ValueError(f"{path}:{number}: manifest split must be train, val or test")
                if split_source == "generated" and assigned:
                    raise ValueError("Existing split assignments found: use split_source: manifest to preserve them")
                records.append(Record(values["id"], str((path.parent / values["image"]).resolve()),
                                      str((path.parent / values["mask"]).resolve()), values["group"], domain, assigned))
    else:
        if not c["independent_images"]:
            raise ValueError("Provide group CSV or explicitly assert independent_images: true")
        def index(root, suffix):
            root = Path(root).resolve()
            result = {}
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() not in c["extensions"]:
                    continue
                stem = p.stem
                if suffix and not stem.endswith(suffix):
                    continue
                name = stem[:-len(suffix)] if suffix else stem
                key = str(p.relative_to(root).parent / name).replace("\\", "/")
                if key in result:
                    raise ValueError(f"Duplicate pairing key {key}")
                result[key] = str(p)
            return result
        images, masks = index(c["images"], c["image_suffix"]), index(c["masks"], c["mask_suffix"])
        if images.keys() != masks.keys():
            raise ValueError(f"Unpaired images/masks: {images.keys() ^ masks.keys()}")
        records = [Record(k, v, masks[k], k, "default") for k, v in images.items()]
    if not records or len({r.id for r in records}) != len(records):
        raise ValueError("Empty dataset or duplicate IDs")
    return sorted(records, key=lambda r: r.id)


def read_mask(path, cfg):
    c = cfg["dataset"]
    raw = np.array(Image.open(path))
    out = np.full(raw.shape[:2], c["ignore_index"], dtype=np.int64)
    known = np.zeros(out.shape, dtype=bool)
    if raw.ndim == 3:
        if not c["color_map"]:
            raise ValueError("RGB masks require color_map, e.g. {'255,0,0': 1}")
        for color, label in c["color_map"].items():
            match = (raw[..., :3] == np.array([int(v) for v in color.split(",")])).all(-1)
            out[match], known[match] = int(label), True
    else:
        if c["label_map"] is None:
            raise ValueError("Explicit verified label_map required; no guessed raw IDs")
        for value, label in c["label_map"].items():
            match = raw == int(value)
            out[match], known[match] = int(label), True
        for value in c["ignore_values"]:
            match = raw == int(value)
            out[match], known[match] = c["ignore_index"], True
    if not known.all():
        raise ValueError(f"Unknown mask values/colors in {path}")
    valid = out != c["ignore_index"]
    if ((out[valid] < 0) | (out[valid] >= len(c["classes"]))).any():
        raise ValueError("Mapped labels outside class vocabulary")
    return out


def audit(records, cfg):
    histogram = np.zeros(len(cfg["dataset"]["classes"]), dtype=np.int64)
    hashes = {}
    for r in records:
        mask = read_mask(r.mask, cfg)
        with Image.open(r.image) as image:
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"Expected RGB image: {r.image}; explicitly convert/select bands first")
            if image.size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"Image/mask size mismatch: {r.id}")
        digest = file_hash(r.image)
        if digest in hashes and hashes[digest] != r.group:
            raise ValueError("Duplicate image content assigned to different groups")
        hashes[digest] = r.group
        valid = mask != cfg["dataset"]["ignore_index"]
        histogram += np.bincount(mask[valid], minlength=len(histogram))
    return {"samples": len(records), "groups": len({r.group for r in records}), "class_pixels": histogram}


def split_records(records, ratios, seed):
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError("Three positive split ratios summing to one required")
    groups = defaultdict(list)
    for r in records:
        groups[r.group].append(r)
    # Cross-domain groups cannot be split independently; fail instead of leaking.
    if any(len({r.domain for r in rows}) > 1 for rows in groups.values()):
        raise ValueError("A group spans domains; merge related domains before splitting")
    result = {k: [] for k in ("train", "val", "test")}
    if any(r.split for r in records):
        if any(r.split not in result for r in records):
            raise ValueError("All records need valid split assignments when preserving manifest splits")
        if any(len({r.split for r in rows}) != 1 for rows in groups.values()):
            raise ValueError("Scene/group leakage: a group occurs in multiple manifest splits")
        for r in records:
            result[r.split].append(r)
        for domain in sorted({r.domain for r in records}):
            if any(not any(r.domain == domain for r in rows) for rows in result.values()):
                raise ValueError(f"Domain {domain} lacks a train, val or test partition")
        return result
    rng = random.Random(seed)
    for domain in sorted({r.domain for r in records}):
        units = [rows for _, rows in sorted(groups.items()) if rows[0].domain == domain]
        if len(units) < 7:
            raise ValueError(f"Domain {domain} needs >=7 independent groups for 70/15/15")
        rng.shuffle(units)
        n = len(units)
        counts = np.floor(np.array(ratios) * n).astype(int)
        for i in np.argsort(-(np.array(ratios) * n - counts), kind="stable")[:n - sum(counts)]:
            counts[i] += 1
        if min(counts) == 0:
            raise ValueError("Empty split: more independent groups needed")
        start = 0
        for name, count in zip(result, counts):
            result[name].extend(r for unit in units[start:start + count] for r in unit)
            start += count
    return result


def save_splits(path, splits, cfg):
    save_json(path, {"seed": cfg["seed"], "requested_ratios": cfg["dataset"]["split"],
                     "source": "manifest" if any(r.split for rows in splits.values() for r in rows) else "generated",
                     "ratio_unit": "independent groups within each domain; existing manifest assignments preserved",
                     "counts": {k: {"images": len(rows), "groups": len({r.group for r in rows})} for k, rows in splits.items()},
                     "partitions": {k: [dict(asdict(r), image_sha256=file_hash(r.image), mask_sha256=file_hash(r.mask))
                                        for r in rows] for k, rows in splits.items()}})


class SegmentationDataset(Dataset):
    def __init__(self, records, cfg, training=False):
        self.records, self.cfg, self.training = records, cfg, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        c = self.cfg["dataset"]
        image = Image.open(r.image).convert("RGB")
        mask = Image.fromarray(read_mask(r.mask, self.cfg).astype(np.int32), mode="I")
        size = c["size"]
        scale = size / max(image.size)
        w, h = [int(x * scale + 0.5) for x in image.size]
        image = np.array(image.resize((w, h), Image.Resampling.BILINEAR)).copy()
        mask = np.array(mask.resize((w, h), Image.Resampling.NEAREST), dtype=np.int64).copy()
        if self.training and c["augment"]:
            if torch.rand(()) < .5:
                image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
            if torch.rand(()) < .5:
                image, mask = image[::-1].copy(), mask[::-1].copy()
        # SAM normalizes then pads. Padding equals pixel mean before normalization.
        canvas = torch.tensor([123.675, 116.28, 103.53])[:, None, None].expand(3, size, size).clone()
        canvas[:, :h, :w] = torch.from_numpy(image).permute(2, 0, 1).float()
        target = torch.full((size, size), c["ignore_index"], dtype=torch.long)
        target[:h, :w] = torch.from_numpy(mask)
        return {"image": canvas, "mask": target, "id": r.id, "domain": r.domain}


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def loader(records, cfg, training=False):
    runtime = cfg["runtime"]
    workers = runtime["num_workers"]
    return DataLoader(SegmentationDataset(records, cfg, training),
                      batch_size=runtime["batch_size" if training else "eval_batch_size"],
                      shuffle=training, num_workers=workers, pin_memory=runtime["pin_memory"] and torch.cuda.is_available(),
                      persistent_workers=workers > 0, worker_init_fn=worker_seed,
                      generator=torch.Generator().manual_seed(cfg["seed"]),
                      **({"prefetch_factor": 2} if workers else {}))
