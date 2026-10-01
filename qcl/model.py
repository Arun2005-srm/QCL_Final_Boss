from pathlib import Path
import hashlib
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .config import file_hash


class QuantumCircuit(nn.Module):
    def __init__(self, chunk=32, entangle=True, device="default.qubit"):
        super().__init__()
        import pennylane as qml
        self.theta = nn.Parameter(torch.empty(2, 8, 2).uniform_(-.1, .1))
        self.chunk = chunk
        if device != "default.qubit":
            raise ValueError("Primary implementation uses verified default.qubit + torch backprop")
        dev = qml.device(device, wires=8, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(a, theta):
            for q in range(8):
                qml.RY(a[..., q], wires=q)
            for layer in range(2):
                for q in range(8):
                    qml.RX(theta[layer, q, 0], wires=q)
                    qml.RY(theta[layer, q, 1], wires=q)
                if entangle:
                    for q in range(7):
                        qml.CNOT(wires=[q, q + 1])
            return tuple(qml.expval(qml.PauliX(q)) for q in range(8))
        self.circuit = circuit

    def evaluate(self, a, theta):
        return torch.stack(self.circuit(a, theta), -1).to(a.dtype)

    def forward(self, a):
        # Circuit arithmetic stays outside mixed precision. Checkpoint each chunk
        # to avoid retaining every gate state for the full spatial batch.
        with torch.autocast(device_type=a.device.type, enabled=False):
            dtype = self.theta.dtype
            chunks = []
            for part in a.to(dtype).split(self.chunk):
                if self.training and torch.is_grad_enabled():
                    q = checkpoint(self.evaluate, part, self.theta, use_reentrant=False)
                else:
                    q = self.evaluate(part, self.theta)
                chunks.append(q)
            return torch.cat(chunks)


class ResidualAdapter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["model"]
        self.kind, self.grid = c["adapter"], c["grid"]
        self.register_buffer("mean", torch.zeros(256))
        self.register_buffer("components", torch.zeros(8, 256))
        self.register_buffer("alpha", torch.tensor(0.))
        self.register_buffer("initialized", torch.tensor(False))
        if self.kind == "none":
            self.core = nn.Identity()
        elif self.kind in {"quantum", "frozen_quantum", "no_entanglement"}:
            self.core = QuantumCircuit(c["quantum_chunk"], self.kind != "no_entanglement" and c["entangle"], c["quantum_device"])
            if self.kind == "frozen_quantum":
                self.core.requires_grad_(False)
        elif self.kind == "sine":
            self.core = Sinusoidal()
        elif self.kind == "mlp":
            self.core = nn.Sequential(nn.Linear(8, c["rank"]), nn.GELU(), nn.Linear(c["rank"], 8), nn.Tanh())
        elif self.kind == "classical":
            self.core = nn.Sequential(nn.Linear(256, c["rank"]), nn.GELU(), nn.Linear(c["rank"], 256, bias=False))
        else:
            raise ValueError(f"Unknown adapter {self.kind}")
        if self.kind not in {"none", "classical"}:
            self.out = nn.Linear(8, 256, bias=False)
            nn.init.orthogonal_(self.out.weight)

    def branch(self, features):
        x = F.adaptive_avg_pool2d(features.float(), self.grid).permute(0, 2, 3, 1)
        if self.kind == "none":
            return torch.zeros_like(features)
        if self.kind == "classical":
            out = self.core(x)
        else:
            z = F.linear(x - self.mean, self.components)
            a = math.pi / 2 * z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            q = self.core(a.reshape(-1, 8)).reshape(*a.shape)
            out = self.out(q)
        return F.interpolate(out.permute(0, 3, 1, 2), features.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, features):
        if not bool(self.initialized):
            raise RuntimeError("Fit training-only PCA and calibrate residual before forward")
        return features + self.alpha * self.branch(features)

    @torch.no_grad()
    def constrain(self):
        if hasattr(self, "out"):
            sigma = torch.linalg.svdvals(self.out.weight.float())[0]
            self.out.weight.div_(sigma.clamp_min(1))


class Sinusoidal(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.phase = nn.Parameter(torch.zeros(8))
        self.scale = nn.Parameter(torch.ones(8))
        self.bias = nn.Parameter(torch.zeros(8))

    def forward(self, a):
        return torch.tanh(self.scale * torch.sin(self.weight * a + self.phase) + self.bias)


class SegmentationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        c = cfg["model"]
        self.synthetic = c["encoder"] == "synthetic"
        if self.synthetic:
            # Test-only deterministic fixed encoder; never a fallback for missing SAM.
            self.encoder = nn.Conv2d(3, 256, 1, bias=False)
            self.encoder.weight.data.copy_(torch.linspace(-1, 1, 256 * 3).reshape(256, 3, 1, 1))
            self.encoder_hash = "synthetic-v1"
        else:
            from segment_anything import sam_model_registry
            if not Path(c["checkpoint"]).is_file():
                raise FileNotFoundError("Provide the SAM ViT-B checkpoint; no random encoder fallback")
            sam = sam_model_registry["vit_b"](checkpoint=c["checkpoint"])
            self.encoder = sam.image_encoder
            self.encoder_hash = file_hash(c["checkpoint"])
        self.encoder.requires_grad_(False).eval()
        self.adapter = ResidualAdapter(cfg)
        self.decoder = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.GroupNorm(16, 128), nn.GELU(),
                                     nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
                                     nn.Conv2d(64, len(cfg["dataset"]["classes"]), 1))
        # Paired methods receive identical decoder initialization even when their
        # adapter constructors consume different numbers of random draws.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"] + 1000)
            for layer in self.decoder.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, images):
        normalized = (images - self.pixel_mean) / self.pixel_std
        if self.synthetic:
            normalized = F.adaptive_avg_pool2d(normalized, 8)
        enabled = images.device.type == "cuda" and self.cfg["runtime"]["amp"]
        # Explicit encoder policy is identical during PCA, cache fill and training.
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=enabled):
            return self.encoder(normalized).float()

    def forward_features(self, features, size=None):
        adapted = self.adapter(features.float())
        low = self.decoder(adapted)
        size = size or (self.cfg["dataset"]["size"],) * 2
        return F.interpolate(low, size, mode="bilinear", align_corners=False), low, adapted

    def forward(self, images):
        return self.forward_features(self.encode(images))


class FeatureStore:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.root = Path(cfg["model"]["cache_dir"]) if cfg["model"]["cache_dir"] else None
        if self.root and cfg["dataset"]["augment"]:
            raise ValueError("Online image augmentation requires cache_dir: null; feature flips are not SAM image flips")

    @torch.no_grad()
    def get(self, batch, device):
        images = batch["image"].to(device, non_blocking=True)
        if self.root is None:
            return self.model.encode(images)
        # Content key includes the actual preprocessed image, checkpoint and dtype.
        rows = []
        for image in images:
            precision = "bf16" if device.type == "cuda" and self.cfg["runtime"]["amp"] else "fp32"
            digest = hashlib.sha256(b"sam-preprocess-v1:" + precision.encode() + image.cpu().numpy().tobytes()).hexdigest()
            path = self.root / self.model.encoder_hash / (digest + ".pt")
            if path.exists():
                feature = torch.load(path, map_location="cpu", weights_only=True)
            else:
                feature = self.model.encode(image[None])[0].half().cpu()
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                torch.save(feature, temporary)
                temporary.replace(path)
            rows.append(feature)
        return torch.stack(rows).to(device=device, dtype=torch.float32)


@torch.no_grad()
def initialize_adapter(model, store, training_loader, cfg, device):
    adapter = model.adapter
    if adapter.kind == "none":
        adapter.initialized.fill_(True)
        return {"alpha": 0.0}
    tokens, calibration = [], []
    count = 0
    generator = torch.Generator().manual_seed(cfg["seed"])
    for batch in training_loader:
        features = store.get(batch, device)
        masks = batch["mask"].to(device)
        valid = F.interpolate((masks != cfg["dataset"]["ignore_index"]).float()[:, None], features.shape[-2:], mode="area")[:, 0] > .999
        x = features.permute(0, 2, 3, 1)[valid].cpu()
        remaining = cfg["model"]["pca_tokens"] - count
        if remaining > 0 and len(x):
            x = x[torch.randperm(len(x), generator=generator)[:remaining]]
            tokens.append(x)
            count += len(x)
        for f, v in zip(features, valid):
            if len(calibration) < 32:
                calibration.append((f.cpu(), v.cpu()))
        if count >= cfg["model"]["pca_tokens"] and len(calibration) >= 32:
            break
    if count < 8:
        raise ValueError("Too few valid first-domain training feature tokens for PCA")
    x = torch.cat(tokens).double()
    mean = x.mean(0)
    centered = x - mean
    _, eigvec = torch.linalg.eigh(centered.T @ centered / max(1, len(x) - 1))
    components = eigvec[:, -8:].flip(1).T
    signs = components[torch.arange(8), components.abs().argmax(1)].sign()
    components *= signs[:, None]
    adapter.mean.copy_(mean.float())
    adapter.components.copy_(components.float())
    sum_f, sum_b, n = 0., 0., 0
    for f, valid in calibration:
        f, valid = f.to(device)[None], valid.to(device)
        b = adapter.branch(f)
        sum_f += f[0, :, valid].square().sum().item()
        sum_b += b[0, :, valid].square().sum().item()
        n += int(valid.sum()) * 256
    s, r0 = math.sqrt(sum_f / n), math.sqrt(sum_b / n)
    adapter.alpha.fill_(.01 * s / max(r0, 1e-6))
    adapter.initialized.fill_(True)
    return {"pca_tokens": count, "calibration_tiles": len(calibration), "feature_rms": s, "r0": r0,
            "alpha": adapter.alpha.item(), "relative_bound": adapter.alpha.item() * math.sqrt(8) / (16 * s)
            if hasattr(adapter, "out") else None}
