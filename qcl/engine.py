from pathlib import Path
import contextlib
import json
import math
import platform
import random
import time
import importlib.metadata
import numpy as np
import torch
import torch.nn.functional as F
from .config import seed_all, save_json, file_hash
from .data import discover, audit, split_records, save_splits, loader, Record
from .model import SegmentationModel, FeatureStore, initialize_adapter, QuantumCircuit
from .metrics import Metrics, segmentation_loss
from .replay import ReplayBuffer
from .reporting import (training_plots, metric_plots, prediction_plot, PrioritySample,
                        probability_plots, tsne_plot, continual_plot, distribution_plots)


def runtime_device(cfg):
    value = cfg["runtime"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def amp_context(cfg, device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" and cfg["runtime"]["amp"] else contextlib.nullcontext()


def metric_accumulator(cfg):
    return Metrics(cfg["dataset"]["classes"], cfg["dataset"]["ignore_index"], cfg["evaluation"]["boundary_fraction"])


def environment(device):
    result = {"python": platform.python_version(), "platform": platform.platform(), "device": str(device),
              "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "pennylane", "numpy", "scikit-learn")}}
    if device.type == "cuda":
        result.update(gpu=torch.cuda.get_device_name(device), memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    return result


@torch.no_grad()
def evaluate(model, store, batches, cfg, device, report_dir=None):
    model.eval()
    metrics = metric_accumulator(cfg)
    loss_sum, count, elapsed = 0., 0, time.perf_counter()
    probability = PrioritySample(cfg["evaluation"]["sampled_pixels"], cfg["seed"])
    embedding = PrioritySample(cfg["evaluation"]["tsne_points"], cfg["seed"]+1)
    per_image, plots = [], 0
    ignore = cfg["dataset"]["ignore_index"]
    if report_dir:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
    for batch in batches:
        target = batch["mask"].to(device, non_blocking=True)
        with amp_context(cfg, device):
            feature = store.get(batch, device)
            logits, _, adapted = model.forward_features(feature)
        loss = segmentation_loss(logits, target, ignore, cfg["training"]["dice_weight"])
        logits = logits.float()
        pred = logits.argmax(1).cpu().numpy()
        truth = target.cpu().numpy()
        metrics.update(pred, truth)
        n = len(target)
        loss_sum += loss.item()*n
        count += n
        if report_dir:
            probs = logits.softmax(1).cpu().numpy()
            low_labels = F.interpolate(target[:, None].float(), adapted.shape[-2:], mode="nearest")[:, 0].long().cpu().numpy()
            vectors = adapted.float().permute(0, 2, 3, 1).cpu().numpy()
            for i in range(n):
                one = metric_accumulator(cfg)
                one.update(pred[i], truth[i])
                per_image.append({"id": batch["id"][i], "domain": batch["domain"][i], **one.compute()})
                valid = truth[i] != ignore
                probability.add(probs[i].transpose(1, 2, 0)[valid], truth[i][valid])
                valid_low = low_labels[i] != ignore
                embedding.add(vectors[i][valid_low], low_labels[i][valid_low],
                              np.full(valid_low.sum(), batch["domain"][i]))
                if plots < cfg["evaluation"]["plots"]:
                    prediction_plot(batch["image"][i].numpy(), truth[i], pred[i], probs[i].max(0),
                                    report_dir/f"prediction_{plots:03}.png", ignore, len(cfg["dataset"]["classes"]))
                    plots += 1
    result = metrics.compute()
    result.update(loss=loss_sum/max(count, 1), seconds=time.perf_counter()-elapsed, images=count)
    if report_dir:
        save_json(report_dir/"metrics.json", result)
        save_json(report_dir/"per_image_metrics.json", per_image)
        metric_plots(result, report_dir)
        distribution_plots(per_image, cfg["dataset"]["classes"], report_dir)
        probability_plots(probability, cfg["dataset"]["classes"], report_dir)
        tsne_plot(embedding, cfg["dataset"]["classes"], report_dir, cfg["seed"], cfg["evaluation"]["tsne_perplexity"])
    return result


def make_optimizer(model, cfg):
    quantum_ids = {id(p) for module in model.modules() if isinstance(module, QuantumCircuit) for p in module.parameters()}
    classical = [p for p in model.parameters() if p.requires_grad and id(p) not in quantum_ids]
    quantum = [p for p in model.parameters() if p.requires_grad and id(p) in quantum_ids]
    groups = [{"params": classical, "lr": cfg["training"]["learning_rate"]}]
    if quantum:
        groups.append({"params": quantum, "lr": cfg["training"]["quantum_learning_rate"]})
    return torch.optim.Adam(groups, betas=(.9, .999), eps=1e-8)


def train_epoch(model, store, batches, optimizer, replay, cfg, device, remaining_updates):
    model.train()
    metrics = metric_accumulator(cfg)
    total_loss, count, updates, micro = 0., 0, 0, 0
    accumulation = cfg["runtime"]["accumulation"]
    ignore = cfg["dataset"]["ignore_index"]
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    residual_log = []
    for idx, batch in enumerate(batches):
        if updates >= remaining_updates:
            break
        target = batch["mask"].to(device, non_blocking=True)
        if not (target != ignore).any():
            continue
        with amp_context(cfg, device):
            features = store.get(batch, device)
            logits, low, adapted = model.forward_features(features)
            current_loss = segmentation_loss(logits, target, ignore, cfg["training"]["dice_weight"])
            loss = current_loss
            old = replay.sample(cfg["continual"]["replay_batch"], device)
            if old:
                rf, rm, rl = old
                old_logits, old_low, _ = model.forward_features(rf)
                replay_loss = segmentation_loss(old_logits, rm, ignore, cfg["training"]["dice_weight"])
                valid = F.interpolate((rm != ignore).float()[:, None], old_low.shape[-2:], mode="area")
                distill = ((old_low.float()-rl).square()*valid).sum()/(valid.sum()*old_low.shape[1]).clamp_min(1)
                loss = loss + cfg["continual"]["replay_weight"]*replay_loss + cfg["continual"]["distill_weight"]*distill
        # Accumulate unscaled gradients, then divide by the actual number of
        # microbatches. This handles a final partial group and skipped void batches.
        loss.backward()
        micro += 1
        for i, key in enumerate(batch["id"]):
            replay.add(key, features[i], target[i], low[i])
        metrics.update(logits.detach().argmax(1).cpu().numpy(), target.cpu().numpy())
        n = len(target)
        # Display current-domain segmentation objective, excluding replay penalties.
        total_loss += current_loss.item()*n
        count += n
        if micro == accumulation:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(micro)
            optimizer.step()
            model.adapter.constrain()
            optimizer.zero_grad(set_to_none=True)
            micro = 0
            updates += 1
        if idx % 50 == 0:
            with torch.no_grad():
                ratio = ((adapted.float()-features).square().mean().sqrt()/features.square().mean().sqrt().clamp_min(1e-8)).item()
                residual_log.append({"batch": idx, "residual_feature_rms_ratio": ratio})
    if micro and updates < remaining_updates:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(micro)
        optimizer.step()
        model.adapter.constrain()
        optimizer.zero_grad(set_to_none=True)
        updates += 1
    result = metrics.compute()
    result.update(loss=total_loss/max(1, count), images=count, updates=updates,
                  seconds=time.perf_counter()-started, residual_log=residual_log)
    return result


def checkpoint_state(model, optimizer, replay, cfg, history, matrix, stage, epoch, updates, generator_state, best):
    return {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("encoder.")},
            "encoder_hash": model.encoder_hash, "split_hash": file_hash(Path(cfg["output"])/"splits.json"),
            "optimizer": optimizer.state_dict(), "replay": replay.state_dict(),
            "config": cfg, "history": history, "matrix": matrix, "stage": stage, "epoch": epoch,
            "updates": updates, "loader_rng": generator_state, "best": best,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def load_model_state(model, state):
    if state["encoder_hash"] != model.encoder_hash:
        raise ValueError("Checkpoint encoder hash mismatch")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected or any(not key.startswith("encoder.") for key in missing):
        raise ValueError(f"Incompatible model state: {missing}, {unexpected}")


def train(cfg, resume=None):
    if cfg["runtime"]["accumulation"] < 1:
        raise ValueError("accumulation must be positive")
    if cfg["dataset"]["ignore_index"] != 255 or len(cfg["dataset"]["classes"]) > 255:
        raise ValueError("Byte-budgeted replay requires ignore_index=255 and <=255 classes")
    seed_all(cfg["seed"])
    root = Path(cfg["output"])
    if (root/"splits.json").exists() and not resume:
        raise FileExistsError("Run directory exists; use --resume or a new output path")
    root.mkdir(parents=True, exist_ok=True)
    records = discover(cfg)
    save_json(root/"data_audit.json", audit(records, cfg))
    splits = split_records(records, cfg["dataset"]["split"], cfg["seed"])
    if resume:
        previous = json.loads((root/"splits.json").read_text())
        for split, rows in previous["partitions"].items():
            if [r.id for r in splits[split]] != [r["id"] for r in rows]:
                raise ValueError("Resume split mismatch")
            for row in rows:
                if any(file_hash(row[key]) != row[key+"_sha256"] for key in ("image", "mask")):
                    raise ValueError("Dataset changed since checkpoint")
    else:
        save_splits(root/"splits.json", splits, cfg)
    device = runtime_device(cfg)
    save_json(root/"environment.json", environment(device))
    save_json(root/"config.json", cfg)
    domains = cfg["continual"]["domains"] or sorted({r.domain for r in records})
    if set(domains) != {r.domain for r in records} or len(set(domains)) != len(domains):
        raise ValueError("continual.domains must list every domain exactly once")
    model = SegmentationModel(cfg).to(device)
    store = FeatureStore(model, cfg)
    optimizer = make_optimizer(model, cfg)
    replay = ReplayBuffer(cfg["continual"]["replay_mib"], cfg["seed"])
    history, matrix, stage_start, epoch_start, saved_updates, best = [], [], 0, 0, 0, -math.inf
    state = None
    if resume:
        # Only load checkpoints produced by this project from trusted storage.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["config"] != cfg:
            raise ValueError("Resume requires identical resolved config")
        if state["split_hash"] != file_hash(root/"splits.json"):
            raise ValueError("Split manifest changed since checkpoint")
        load_model_state(model, state)
        optimizer.load_state_dict(state["optimizer"])
        replay.load_state_dict(state["replay"])
        history, matrix = state["history"], state["matrix"]
        stage_start, epoch_start, saved_updates, best = state["stage"], state["epoch"], state["updates"], state["best"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng"]["cuda"]])
    else:
        first = loader([r for r in splits["train"] if r.domain == domains[0]], cfg)
        save_json(root/"initialization.json", initialize_adapter(model, store, first, cfg, device))
    save_json(root/"model_summary.json", {"total_parameters": sum(p.numel() for p in model.parameters()),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "encoder_hash": model.encoder_hash, "architecture": str(model), "domains": domains})
    for stage in range(stage_start, len(domains)):
        domain = domains[stage]
        batches = loader([r for r in splits["train"] if r.domain == domain], cfg, True)
        if state is not None and stage == stage_start:
            batches.generator.set_state(state["loader_rng"].cpu())
        validation = loader([r for r in splits["val"] if r.domain in domains[:stage+1]], cfg)
        updates = saved_updates if stage == stage_start else 0
        first_epoch = epoch_start if stage == stage_start else 0
        steps_per_epoch = math.ceil(len(batches)/cfg["runtime"]["accumulation"])
        planned_epochs = min(cfg["training"]["epochs"], math.ceil(cfg["training"]["max_updates_per_domain"]/steps_per_epoch))
        for epoch in range(first_epoch, planned_epochs):
            remaining = cfg["training"]["max_updates_per_domain"]-updates
            if remaining <= 0:
                break
            tr = train_epoch(model, store, batches, optimizer, replay, cfg, device, remaining)
            updates += tr["updates"]
            val = evaluate(model, store, validation, cfg, device)
            history.append({"domain": domain, "epoch": epoch+1, "domain_updates": updates, "train": tr, "val": val})
            print(f"{domain} ep[{epoch+1}/{planned_epochs}] updates={updates} | "
                  f"train acc={tr['accuracy']:.4f} loss={tr['loss']:.4f} Dice={tr['dice']:.4f} IoU={tr['iou']:.4f} mIoU={tr['miou']:.4f} bIoU={tr['biou']:.4f} | "
                  f"val acc={val['accuracy']:.4f} loss={val['loss']:.4f} "
                  f"Dice={val['dice']:.4f} IoU={val['iou']:.4f} mIoU={val['miou']:.4f} bIoU={val['biou']:.4f}", flush=True)
            print("train class IoU: " + " | ".join(f"{k}={v['iou']:.4f}" for k, v in tr["classwise"].items()), flush=True)
            print("val class IoU: " + " | ".join(f"{k}={v['iou']:.4f}" for k, v in val["classwise"].items()), flush=True)
            save_json(root/"history.json", history)
            training_plots(history, root)
            improved = val["miou"] > best
            best = max(best, val["miou"])
            payload = checkpoint_state(model, optimizer, replay, cfg, history, matrix, stage, epoch+1, updates, batches.generator.get_state(), best)
            torch.save(payload, root/"last.pt")
            if improved:
                torch.save(payload, root/f"best_stage_{stage:02}.pt")
        row = [evaluate(model, store, loader([r for r in splits["val"] if r.domain == d], cfg), cfg, device)["miou"]
               if i <= stage else float("nan") for i, d in enumerate(domains)]
        matrix.append(row)
        a = np.array(matrix)
        forgetting = [np.nanmax(a[:, i])-a[-1, i] for i in range(stage)]
        save_json(root/"continual_metrics.json", {"domains": domains, "validation_miou_matrix": matrix,
                  "average_seen_miou": np.nanmean(row), "mean_forgetting_previous": np.mean(forgetting) if forgetting else None,
                  "backward_transfer": np.mean([a[-1, i]-a[i, i] for i in range(stage)]) if stage else None})
        continual_plot(matrix, domains, root)
        payload = checkpoint_state(model, optimizer, replay, cfg, history, matrix, stage+1, 0, 0, batches.generator.get_state(), -math.inf)
        torch.save(payload, root/f"stage_{stage:02}.pt")
        torch.save(payload, root/"last.pt")
        best = -math.inf
    torch.save(payload if 'payload' in locals() else state, root/"final.pt")
    save_json(root/"run_status.json", {"status": "training completed", "scientific_gate": "not evaluated",
              "domain_updates": {d: max((r["domain_updates"] for r in history if r["domain"] == d), default=0) for d in domains},
              "note": "Epoch limit can stop before 2000 updates. Only full-budget runs qualify for protocol comparisons."})
    return root/"final.pt"


def test(checkpoint_path, output=None):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    seed_all(cfg["seed"])
    device = runtime_device(cfg)
    model = SegmentationModel(cfg).to(device)
    load_model_state(model, state)
    root = Path(cfg["output"])
    if file_hash(root/"splits.json") != state["split_hash"]:
        raise ValueError("Split manifest changed since checkpoint")
    split = json.loads((root/"splits.json").read_text())
    records = []
    for row in split["partitions"]["test"]:
        for key in ("image", "mask"):
            if file_hash(row[key]) != row[key+"_sha256"]:
                raise ValueError("Test dataset changed since split creation")
        records.append(Record(**{k: row[k] for k in ("id", "image", "mask", "group", "domain")}))
    destination = Path(output) if output else root/"test"/Path(checkpoint_path).stem
    if destination.exists():
        raise FileExistsError("Test report already exists; choose --output to preserve previous results")
    store = FeatureStore(model, cfg)
    result = evaluate(model, store, loader(records, cfg), cfg, device, destination)
    per_domain = {d: evaluate(model, store, loader([r for r in records if r.domain == d], cfg), cfg, device)
                  for d in sorted({r.domain for r in records})}
    save_json(destination/"domain_metrics.json", per_domain)
    save_json(destination/"provenance.json", {"checkpoint_sha256": file_hash(checkpoint_path),
              "split_sha256": file_hash(root/"splits.json"), "environment": environment(device)})
    print(json.dumps({k: v for k, v in result.items() if k not in {"classwise", "confusion_matrix"}}, indent=2))
    return result
