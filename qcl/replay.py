import random
import torch


class ReplayBuffer:
    """Byte-capped reservoir over unique arrivals, with frozen insertion logits."""
    def __init__(self, mib, seed=42):
        self.budget = int(mib * 1024**2)
        self.rng = random.Random(seed)
        self.items, self.seen = [], set()
        self.record_bytes = None

    def add(self, key, feature, mask, logits):
        if key in self.seen:
            return
        self.seen.add(key)
        item = (feature.detach().half().cpu(), mask.detach().to(torch.uint8).cpu(), logits.detach().half().cpu())
        size = sum(x.numel()*x.element_size() for x in item) + 512
        if self.record_bytes is None:
            self.record_bytes = size
        if size != self.record_bytes:
            raise ValueError("Replay records must have fixed shape")
        capacity = self.budget // size
        if not capacity:
            return
        if len(self.items) < capacity:
            self.items.append(item)
        else:
            index = self.rng.randrange(len(self.seen))
            if index < capacity:
                self.items[index] = item

    def sample(self, count, device):
        if not self.items or count == 0:
            return None
        rows = self.rng.sample(self.items, min(count, len(self.items)))
        return tuple(torch.stack([r[i] for r in rows]).to(device).float() if i != 1 else
                     torch.stack([r[i] for r in rows]).to(device).long() for i in range(3))

    def state_dict(self):
        return {"budget": self.budget, "items": self.items, "seen": sorted(self.seen),
                "record_bytes": self.record_bytes, "rng": self.rng.getstate()}

    def load_state_dict(self, state):
        self.budget, self.items = state["budget"], state["items"]
        self.seen, self.record_bytes = set(state["seen"]), state["record_bytes"]
        self.rng.setstate(state["rng"])
