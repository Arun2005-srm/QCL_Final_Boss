from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .config import save_json


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_plots(history, root):
    if not history:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, ("loss", "accuracy", "dice", "iou", "miou", "biou")):
        for split in ("train", "val"):
            ax.plot(range(1, len(history)+1), [r[split][metric] for r in history], label=split)
        ax.set(title=metric, xlabel="Logged epoch (across domains)")
        ax.legend()
    finish(fig, Path(root)/"training_curves.png")
    rows = []
    for entry in history:
        row = {k: entry[k] for k in ("domain", "epoch", "domain_updates")}
        for split in ("train", "val"):
            for key, value in entry[split].items():
                if np.isscalar(value):
                    row[f"{split}_{key}"] = value
            for name, scores in entry[split]["classwise"].items():
                row[f"{split}_iou_{name}"] = scores["iou"]
        rows.append(row)
    with (Path(root)/"history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution_plots(per_image, names, root):
    root = Path(root)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key in zip(axes, ("miou", "dice", "biou")):
        values = [r[key] for r in per_image if np.isfinite(r[key])]
        ax.hist(values, bins=min(20, max(1, len(values))), range=(0, 1))
        ax.set(title=f"Per-image {key}", xlabel="Score", ylabel="Images")
    finish(fig, root/"per_image_distributions.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    scores = [[r["classwise"][name]["iou"] for r in per_image if np.isfinite(r["classwise"][name]["iou"])] for name in names]
    ax.boxplot([x if x else [np.nan] for x in scores], tick_labels=names)
    ax.set(ylabel="Per-image class IoU", ylim=(0, 1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    finish(fig, root/"classwise_iou_distributions.png")


def metric_plots(metrics, root):
    root = Path(root)
    names = list(metrics["classwise"])
    cm = np.array(metrics["confusion_matrix"])
    for normalized in (False, True):
        matrix = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalized else cm
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
               xlabel="Predicted", ylabel="Ground truth", title="Normalized confusion" if normalized else "Pixel confusion")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        finish(fig, root / ("confusion_normalized.png" if normalized else "confusion_counts.png"))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    for offset, key in enumerate(("iou", "dice", "biou")):
        vals = [metrics["classwise"][n][key] for n in names]
        ax.bar(x+offset*.25, vals, .25, label=key)
    ax.set(xticks=x+.25, xticklabels=names, ylim=(0, 1), ylabel="Score")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    finish(fig, root/"classwise_scores.png")
    with (root/"classwise_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class"]+list(metrics["classwise"][names[0]]))
        writer.writeheader()
        for name in names:
            writer.writerow({"class": name, **metrics["classwise"][name]})


def prediction_plot(image, target, pred, confidence, path, ignore, classes):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    valid = target != ignore
    target = np.ma.masked_where(~valid, target)
    pred = np.ma.masked_where(~valid, pred)
    axes[0].imshow(np.clip(image.transpose(1, 2, 0)/255, 0, 1))
    axes[1].imshow(target, vmin=0, vmax=classes-1, cmap="tab20")
    axes[2].imshow(pred, vmin=0, vmax=classes-1, cmap="tab20")
    axes[3].imshow(np.ma.masked_where(~valid, pred != target), cmap="Reds", vmin=0, vmax=1)
    axes[4].imshow(np.ma.masked_where(~valid, confidence), cmap="viridis", vmin=0, vmax=1)
    for ax, title in zip(axes, ("Image", "Ground truth", "Prediction", "Error", "Confidence")):
        ax.set_title(title)
        ax.axis("off")
    finish(fig, path)


class PrioritySample:
    """Uniform bounded reservoir via independent random priorities."""
    def __init__(self, capacity, seed):
        self.capacity, self.rng = capacity, np.random.default_rng(seed)
        self.keys, self.arrays = np.empty(0), None

    def add(self, *arrays):
        if not len(arrays[0]) or self.capacity <= 0:
            return
        keys = self.rng.random(len(arrays[0]))
        take = np.argpartition(keys, min(self.capacity, len(keys))-1)[:self.capacity]
        keys, arrays = keys[take], [a[take] for a in arrays]
        if self.arrays is not None:
            keys = np.concatenate((self.keys, keys))
            arrays = [np.concatenate((old, new)) for old, new in zip(self.arrays, arrays)]
        take = np.argsort(keys)[:self.capacity]
        self.keys, self.arrays = keys[take], [a[take] for a in arrays]


def probability_plots(sample, names, root):
    if sample.arrays is None:
        save_json(Path(root)/"probability_status.json", {"status": "no valid sampled pixels"})
        return
    probabilities, labels = sample.arrays
    root = Path(root)
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    scores = {}
    for c, name in enumerate(names):
        y = labels == c
        if y.all() or not y.any():
            scores[name] = {"roc_auc": None, "average_precision": None, "reason": "missing positive or negative samples"}
            continue
        fpr, tpr, _ = roc_curve(y, probabilities[:, c])
        precision, recall, _ = precision_recall_curve(y, probabilities[:, c])
        ax_roc.plot(fpr, tpr, label=name)
        ax_pr.plot(recall, precision, label=name)
        scores[name] = {"roc_auc": roc_auc_score(y, probabilities[:, c]), "average_precision": average_precision_score(y, probabilities[:, c])}
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="Sampled pixel ROC (one-vs-rest)")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Sampled pixel precision–recall")
    for ax in (ax_roc, ax_pr):
        if ax.lines:
            ax.legend(fontsize="small")
    finish(fig_roc, root/"roc_sampled.png")
    finish(fig_pr, root/"precision_recall_sampled.png")
    confidence, pred = probabilities.max(1), probabilities.argmax(1)
    correct = pred == labels
    bins = np.minimum((confidence*10).astype(int), 9)
    accuracy, conf, counts = [], [], []
    for b in range(10):
        mask = bins == b
        counts.append(int(mask.sum()))
        accuracy.append(float(correct[mask].mean()) if mask.any() else np.nan)
        conf.append(float(confidence[mask].mean()) if mask.any() else np.nan)
    ece = np.nansum(np.abs(np.array(accuracy)-conf)*np.array(counts))/len(labels)
    brier = ((probabilities-np.eye(len(names))[labels])**2).sum(1).mean()
    nll = -np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-12)).mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(conf, accuracy, "o-")
    ax.set(xlabel="Confidence", ylabel="Accuracy", title="Sampled reliability diagram", xlim=(0, 1), ylim=(0, 1))
    finish(fig, root/"calibration_sampled.png")
    save_json(root/"sampled_probability_metrics.json", {"sample_count": len(labels), "sampling": "uniform valid pixels",
              "per_class": scores, "ece_10_bins": ece, "brier_multiclass": brier, "nll": nll,
              "calibration_bin_counts": counts})


def tsne_plot(sample, names, root, seed, perplexity):
    root = Path(root)
    if sample.arrays is None or len(sample.arrays[0]) < 4:
        save_json(root/"tsne_status.json", {"status": "skipped: fewer than four valid feature tokens"})
        return
    features, labels, domains = sample.arrays
    dimension = min(50, features.shape[1], len(features)-1)
    reduced = PCA(n_components=dimension, random_state=seed).fit_transform(features)
    perplexity = min(float(perplexity), max(1., (len(features)-1)/3))
    xy = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed).fit_transform(reduced)
    for label, values, classes in (("class", labels, list(enumerate(names))),
                                    ("domain", domains, [(x, str(x)) for x in np.unique(domains)])):
        fig, ax = plt.subplots(figsize=(9, 7))
        for value, name in classes:
            mask = values == value
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=7, alpha=.65, label=name)
        ax.set_title(f"t-SNE: adapted SAM spatial features by {label}")
        ax.legend(fontsize="small", markerscale=2)
        finish(fig, root/f"tsne_{label}.png")
    np.savez_compressed(root/"tsne_points.npz", xy=xy, labels=labels, domains=domains)
    save_json(root/"tsne_metadata.json", {"seed": seed, "points": len(features), "perplexity": perplexity,
              "feature": "adapted encoder spatial tokens; nearest-resized ground-truth class",
              "sampling": "uniform valid tokens", "warning": "Exploratory visualization; not a separation or accuracy metric"})


def continual_plot(matrix, domains, root):
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(np.array(matrix, dtype=float), vmin=0, vmax=1, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Validation mIoU")
    ax.set(xticks=range(len(domains)), yticks=range(len(matrix)), xticklabels=domains,
           xlabel="Evaluated domain", ylabel="Training stage", title="Continual retention matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    finish(fig, Path(root)/"continual_matrix.png")
