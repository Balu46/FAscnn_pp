import torch.optim as optim
# -----------------------------
# PolyLR (per-iteration)
# -----------------------------
class PolyLR:
    def __init__(self, optimizer: optim.Optimizer, max_iters: int, power: float = 0.9):
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.power = power
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        t = min(self.step_count, self.max_iters)
        factor = (1.0 - float(t) / float(self.max_iters)) ** self.power
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]

