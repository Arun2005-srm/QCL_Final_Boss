import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from qcl.config import load_config, save_json
from qcl.data import Record, split_records, read_mask, audit, discover
from qcl.metrics import Metrics, segmentation_loss
from qcl.model import QuantumCircuit, SegmentationModel, FeatureStore
from qcl.replay import ReplayBuffer
from qcl.engine import train, test as evaluate_checkpoint


torch.set_num_threads(2)


def tiny_config(tmp_path):
    cfg = load_config(Path(__file__).parents[1]/"configs/default.yaml")
    image_root, mask_root = tmp_path/"images", tmp_path/"masks"
    image_root.mkdir()
    mask_root.mkdir()
    rng = np.random.default_rng(42)
    for i in range(10):
        image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = (image[..., 0] > 127).astype(np.uint8)*255
        Image.fromarray(image).save(image_root/f"{i}.png")
        Image.fromarray(mask).save(mask_root/f"{i}.png")
    cfg["dataset"].update(images=str(image_root), masks=str(mask_root), independent_images=True, size=32)
    cfg["model"].update(encoder="synthetic", grid=2, quantum_chunk=4, cache_dir=None, pca_tokens=64)
    # Keep synthetic test budgets independent of user-tuned training defaults.
    cfg["runtime"].update(device="cpu", amp=False, num_workers=0,
                          batch_size=2, eval_batch_size=2, accumulation=1)
    cfg["training"].update(epochs=1, max_updates_per_domain=1)
    cfg["evaluation"].update(plots=1, sampled_pixels=128, tsne_points=32)
    cfg["continual"].update(replay_mib=.02)
    cfg["output"] = str(tmp_path/"run")
    return cfg


def test_split_group_isolation_and_determinism():
    records = [Record(f"{g}-{i}", "a", "b", str(g), "d") for g in range(20) for i in range(2)]
    a = split_records(records, [.7, .15, .15], 42)
    b = split_records(records, [.7, .15, .15], 42)
    assert a == b
    sets = [{r.group for r in rows} for rows in a.values()]
    assert [len(x) for x in sets] == [14, 3, 3]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]


def test_metrics_absent_class_false_positive_and_ignore():
    meter = Metrics(["a", "b", "absent"])
    meter.update(np.array([[0, 1], [2, 2]]), np.array([[0, 1], [1, 255]]))
    m = meter.compute()
    assert m["accuracy"] == pytest.approx(2/3)
    assert m["classwise"]["absent"]["iou"] == 0
    assert m["miou"] == pytest.approx((1+.5+0)/3)
    assert m["valid_pixels"] == 3
    empty = Metrics(["a", "b"])
    empty.update(np.zeros((2, 2), int), np.full((2, 2), 255))
    assert np.isnan(empty.compute()["miou"])


def test_boundary_identity_and_shift():
    truth = np.zeros((40, 40), dtype=int)
    truth[10:30, 10:30] = 1
    same, shifted = Metrics(["a", "b"]), Metrics(["a", "b"])
    same.update(truth, truth)
    shifted.update(np.roll(truth, 5, axis=0), truth)
    assert same.compute()["biou"] == 1
    assert shifted.compute()["biou"] < 1


def test_loss_keeps_absent_classes_and_all_ignore():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.zeros(2, 8, 8, dtype=torch.long)
    loss = segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad[:, 2].abs().sum() > 0
    assert segmentation_loss(logits, torch.full_like(target, 255)).item() == 0


def test_quantum_chunk_matches_full_gradients():
    torch.manual_seed(42)
    small, full = QuantumCircuit(2).double(), QuantumCircuit(100).double()
    full.load_state_dict(small.state_dict())
    a = torch.randn(7, 8, dtype=torch.float64)
    x, y = small(a), full(a)
    x.square().sum().backward()
    y.square().sum().backward()
    torch.testing.assert_close(x, y, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(small.theta.grad, full.theta.grad, atol=1e-9, rtol=1e-7)
    assert small.theta.grad.abs().sum() > 0


def test_replay_byte_cap_unique_arrivals_and_restore():
    buffer = ReplayBuffer(.01)
    f, m, l = torch.ones(256, 2, 2), torch.zeros(8, 8), torch.ones(3, 2, 2)
    for i in range(20):
        buffer.add(str(i), f*i, m, l*i)
    for i in range(20):
        buffer.add(str(i), f, m, l)
    assert len(buffer.seen) == 20
    assert len(buffer.items)*buffer.record_bytes <= buffer.budget
    restored = ReplayBuffer(.01)
    restored.load_state_dict(buffer.state_dict())
    for x, y in zip(buffer.sample(2, "cpu"), restored.sample(2, "cpu")):
        torch.testing.assert_close(x, y)


def test_unknown_labels_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    path = tmp_path/"unknown.png"
    Image.fromarray(np.full((4, 4), 7, np.uint8)).save(path)
    with pytest.raises(ValueError, match="Unknown"):
        read_mask(path, cfg)


def test_full_train_test_resume_and_reports(tmp_path):
    cfg = tiny_config(tmp_path)
    checkpoint = train(cfg)
    report = evaluate_checkpoint(checkpoint)
    assert np.isfinite(report["loss"])
    root = Path(cfg["output"])
    for name in ("training_curves.png", "continual_metrics.json", "model_summary.json"):
        assert (root/name).is_file()
    for name in ("metrics.json", "tsne_class.png", "tsne_domain.png", "roc_sampled.png",
                 "precision_recall_sampled.png", "calibration_sampled.png", "classwise_metrics.csv"):
        assert (root/"test"/"final"/name).is_file()
    resumed = train(cfg, root/"last.pt")
    state = torch.load(resumed, weights_only=False)
    assert len(state["history"]) == 1
    assert state["stage"] == 1


def test_duplicate_image_group_leakage_rejected(tmp_path):
    cfg = tiny_config(tmp_path)
    records = discover(cfg)
    duplicate = Record("duplicate", records[0].image, records[0].mask, "different-group", "default")
    with pytest.raises(ValueError, match="Duplicate image"):
        audit(records+[duplicate], cfg)


def test_dataset_presets_replace_raw_label_maps():
    root = Path(__file__).parents[1]
    oem = load_config(root/"configs/datasets/openearthmap.yaml")
    landcover = load_config(root/"configs/datasets/landcover_ai.yaml")
    assert set(oem["dataset"]["label_map"]) == set(range(1, 9))
    assert set(landcover["dataset"]["label_map"]) == set(range(5))


def test_two_domain_replay_and_stage_checkpoints(tmp_path):
    import csv
    cfg = tiny_config(tmp_path)
    rows = []
    for domain in ("first", "second"):
        for i in range(10):
            # Separate source groups, unique source contents.
            path = tmp_path/"images"/f"{i}.png"
            image = np.array(Image.open(path)).copy()
            if domain == "second":
                image = np.roll(image, 1, axis=0)
            target_path = tmp_path/"images"/f"{domain}_{i}.png"
            Image.fromarray(image).save(target_path)
            rows.append({"id": f"{domain}_{i}", "image": str(target_path), "mask": str(tmp_path/"masks"/f"{i}.png"),
                         "group": f"{domain}_{i}", "domain": domain})
    manifest = tmp_path/"manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cfg["dataset"]["manifest"] = str(manifest)
    cfg["model"]["adapter"] = "classical"
    cfg["continual"]["domains"] = ["first", "second"]
    checkpoint = train(cfg)
    state = torch.load(checkpoint, weights_only=False)
    assert state["stage"] == 2 and len(state["matrix"]) == 2
    assert len(state["replay"]["seen"]) == 4
    assert state["updates"] == 0  # final checkpoint is ready for the next stage
    assert (Path(cfg["output"])/"stage_00.pt").exists()
    assert np.isfinite(state["matrix"][1][0])


def test_feature_cache_content_key_and_augmentation_guard(tmp_path):
    cfg = tiny_config(tmp_path)
    cfg["model"]["cache_dir"] = str(tmp_path/"cache")
    model = SegmentationModel(cfg)
    store = FeatureStore(model, cfg)
    image = torch.full((1, 3, 32, 32), 100.)
    batch = {"image": image}
    first = store.get(batch, torch.device("cpu"))
    second = store.get(batch, torch.device("cpu"))
    torch.testing.assert_close(first, second)
    assert len(list((tmp_path/"cache").rglob("*.pt"))) == 1
    store.get({"image": image+1}, torch.device("cpu"))
    assert len(list((tmp_path/"cache").rglob("*.pt"))) == 2
    cfg["dataset"]["augment"] = True
    with pytest.raises(ValueError, match="augmentation"):
        FeatureStore(model, cfg)


def test_paired_decoder_initialization(tmp_path):
    cfg = tiny_config(tmp_path)
    q = SegmentationModel(cfg)
    other = copy.deepcopy(cfg)
    other["model"]["adapter"] = "classical"
    c = SegmentationModel(other)
    for a, b in zip(q.decoder.parameters(), c.decoder.parameters()):
        torch.testing.assert_close(a, b)
