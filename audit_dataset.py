import argparse
from pathlib import Path
from qcl.config import load_config, save_json
from qcl.data import discover, audit, split_records, save_splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/audit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    records = discover(cfg)
    result = audit(records, cfg)
    splits = split_records(records, cfg["dataset"]["split"], cfg["seed"])
    result["split_counts"] = {s: {"images": len(rows), "groups": len({r.group for r in rows})} for s, rows in splits.items()}
    save_json(Path(args.output)/"audit.json", result)
    save_splits(Path(args.output)/"splits.json", splits, cfg)
    print(result)
