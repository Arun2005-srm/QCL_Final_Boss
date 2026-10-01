"""Bounded correctness and throughput checks; no scientific gate claim."""
import argparse
from pathlib import Path
import time
import torch
from qcl.config import load_config, seed_all, save_json
from qcl.model import QuantumCircuit, SegmentationModel
from qcl.engine import runtime_device, environment, amp_context
from qcl.metrics import segmentation_loss


def circuit_check(device):
    seed_all(42)
    circuit = QuantumCircuit(chunk=8).to(device).double()
    raw = torch.randn(64, 8, device=device, dtype=torch.float64)
    angles = torch.pi/2 * raw/raw.norm(dim=1, keepdim=True)
    theta = circuit.theta.detach().requires_grad_()
    jac = torch.autograd.functional.jacobian(lambda p: circuit.evaluate(angles, p), theta, vectorize=False)
    singular = torch.linalg.svdvals(jac.reshape(512, 32))
    output = circuit(angles)
    output.square().sum().backward()
    return {"seed": 42, "inputs": 64, "angles": angles.detach().cpu().tolist(), "theta": theta.detach().cpu().tolist(),
            "jacobian_shape": [512, 32], "singular_values": singular.detach().cpu().tolist(),
            "rank_1e-8": int((singular > singular[0]*1e-8).sum()),
            "rank_1e-6": int((singular > singular[0]*1e-6).sum()),
            "gradient_finite": bool(torch.isfinite(circuit.theta.grad).all()),
            "gradient_norm": circuit.theta.grad.norm().item()}


def profile(cfg, output, repetitions=5, rank_check=False):
    seed_all(cfg["seed"])
    device = runtime_device(cfg)
    model = SegmentationModel(cfg).to(device)
    # Synthetic features profile adapter/decoder only, not accuracy or PCA fitting.
    size = 8 if model.synthetic else 64
    batch = cfg["runtime"]["batch_size"]
    features = torch.randn(batch, 256, size, size, device=device)
    components = torch.linalg.qr(torch.randn(256, 8, device=device)).Q.T
    model.adapter.components.copy_(components)
    model.adapter.alpha.fill_(.01)
    model.adapter.initialized.fill_(True)
    target = torch.randint(len(cfg["dataset"]["classes"]), (batch, cfg["dataset"]["size"], cfg["dataset"]["size"]), device=device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(repetitions+2):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp_context(cfg, device):
            logits, _, _ = model.forward_features(features)
            loss = segmentation_loss(logits, target)
        loss.backward()
        opt.step()
        model.adapter.constrain()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= 2:
            times.append(time.perf_counter()-started)
    result = {"environment": environment(device), "batch_size": batch, "quantum_chunk": cfg["model"]["quantum_chunk"],
              "seconds_per_update": times, "mean_seconds": sum(times)/len(times),
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "scope": "adapter+decoder optimizer update on synthetic cached features; excludes encoder, IO, metrics and validation",
              "scientific_gate": "not evaluated"}
    if rank_check:
        result["circuit_numerics"] = circuit_check(device)
    save_json(output, result)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", default="outputs/stage0/profile.json")
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--rank-check", action="store_true")
    args = p.parse_args()
    profile(load_config(args.config), args.output, args.repetitions, args.rank_check)
