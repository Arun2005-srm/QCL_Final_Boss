import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion


def divide(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast_shapes(a.shape, b.shape), np.nan), where=b != 0)


def mean_valid(x):
    x = np.asarray(x)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


class Metrics:
    def __init__(self, classes, ignore=255, boundary_fraction=.02):
        self.classes, self.ignore, self.fraction = classes, ignore, boundary_fraction
        self.cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
        self.bi = np.zeros(len(classes), dtype=np.int64)
        self.bu = self.bi.copy()

    def update(self, pred, target):
        pred, target = np.asarray(pred), np.asarray(target)
        if target.ndim == 2:
            pred, target = pred[None], target[None]
        k = len(self.classes)
        for p, t in zip(pred, target):
            valid = t != self.ignore
            if not valid.any():
                continue
            self.cm += np.bincount(k * t[valid] + p[valid], minlength=k*k).reshape(k, k)
            # Boundary width derives from the valid image extent, not SAM padding.
            ys, xs = np.where(valid)
            radius = max(1, round(self.fraction * np.hypot(ys.max()-ys.min()+1, xs.max()-xs.min()+1)))
            # Exclude boundaries against void/padding; image border itself is retained.
            safe = binary_erosion(valid, iterations=radius, border_value=1)
            for c in range(k):
                a, b = (p == c) & valid, (t == c) & valid
                ab = a & ~binary_erosion(a, iterations=radius, border_value=0) & safe
                bb = b & ~binary_erosion(b, iterations=radius, border_value=0) & safe
                self.bi[c] += (ab & bb).sum()
                self.bu[c] += (ab | bb).sum()

    def compute(self):
        cm = self.cm.astype(float)
        tp, support, predicted = cm.diagonal(), cm.sum(1), cm.sum(0)
        total = cm.sum()
        fp, fn = predicted-tp, support-tp
        tn = total-tp-fp-fn
        iou = divide(tp, tp+fp+fn)
        dice = divide(2*tp, 2*tp+fp+fn)
        precision, recall = divide(tp, predicted), divide(tp, support)
        specificity = divide(tn, tn+fp)
        expected = float((support*predicted).sum() / total**2) if total else float("nan")
        acc = float(tp.sum()/total) if total else float("nan")
        # Multiclass Matthews coefficient from the complete confusion matrix.
        denom = np.sqrt((total**2-(predicted**2).sum())*(total**2-(support**2).sum()))
        mcc = float(divide(total*tp.sum()-(support*predicted).sum(), denom))
        biou = divide(self.bi, self.bu)
        return {"accuracy": acc, "dice": mean_valid(dice), "miou": mean_valid(iou),
                "iou": float(divide(tp.sum(), (tp+fp+fn).sum())),
                "biou": mean_valid(biou), "precision_macro": mean_valid(precision),
                "recall_macro": mean_valid(recall), "specificity_macro": mean_valid(specificity),
                "balanced_accuracy": mean_valid(recall), "mcc": mcc,
                "kappa": float(divide(acc-expected, 1-expected)),
                "frequency_weighted_iou": float(np.nansum(divide(support, total)*iou)) if total else float("nan"),
                "valid_pixels": int(total), "confusion_matrix": self.cm.tolist(),
                "classwise": {name: {"iou": iou[i], "dice": dice[i], "precision": precision[i],
                                      "recall": recall[i], "specificity": specificity[i], "biou": biou[i],
                                      "support": int(support[i])} for i, name in enumerate(self.classes)}}


def segmentation_loss(logits, target, ignore=255, dice_weight=.5):
    logits = logits.float()
    valid = target != ignore
    if not bool(valid.any()):
        return logits.sum() * 0
    ce = F.cross_entropy(logits, target, ignore_index=ignore)
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float()
    onehot *= valid[:, None]
    prob = logits.softmax(1) * valid[:, None]
    dims = (0, 2, 3)
    present = onehot.sum(dims) > 0
    dice = (2*(prob*onehot).sum(dims)+1e-6)/(prob.sum(dims)+onehot.sum(dims)+1e-6)
    return ce + dice_weight * (1-dice[present].mean())
