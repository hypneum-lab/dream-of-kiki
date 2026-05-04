"""Robertson 2018 sequential-ordering test.

Robertson (Curr Biol 2018) reports that the ORDER of NREM→REM
matters for memory consolidation : sequence-dependent reactivation
patterns determine what gets retained.

Our framework specifies 4 dream operations :
  REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE

Question : does the order in which they fire affect retention on
a continual-learning stream, holding all hyperparameters constant ?

Design :
  - 6 permutations of 4 operations (sample, not all 24)
  - 5 seeds × 6 permutations = 30 cells
  - Synthetic CL stream : 5 tasks × 200 examples each
  - MLP 2x32, MLX backend
  - Metric : average task retention after full stream

Hypothesis (descriptive, not pre-registered) :
  H_RO-A : permutation effect on retention is small (Hedges' g < 0.2 across pairs)
  H_RO-B : a specific order shows a meaningful improvement (g >= 0.4)
  H_RO-C : null effect — retention statistically indistinguishable across orders
"""
import itertools
import json
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def make_cl_stream(n_tasks=5, n_per_task=200, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for t in range(n_tasks):
        center = rng.normal(0, 2.0, size=(2, dim))
        x = np.concatenate([
            center[0] + rng.normal(0, 0.5, size=(n_per_task // 2, dim)),
            center[1] + rng.normal(0, 0.5, size=(n_per_task // 2, dim)),
        ]).astype(np.float32)
        y = np.concatenate([np.zeros(n_per_task // 2), np.ones(n_per_task // 2)]).astype(np.int32)
        # Shift labels by task ID (5 tasks → 10-way classification)
        y = y + 2 * t
        Xs.append(x)
        ys.append(y)
    return Xs, ys


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        return self.fc3(x)


def loss_fn(model, x, y):
    logits = model(x)
    return mx.mean(nn.losses.cross_entropy(logits, y))


def apply_dream_op(model, op_name, replay_buffer, seed):
    """Stub dream operation — small parameter perturbation that mimics op semantics."""
    rng = np.random.default_rng(seed)
    params = dict(model.parameters())
    if op_name == "REPLAY":
        # Replay buffer mini-train (1 step on stored examples)
        if replay_buffer is not None and len(replay_buffer[0]) > 0:
            x_buf, y_buf = replay_buffer
            opt = optim.SGD(learning_rate=0.01)
            grad_fn = nn.value_and_grad(model, loss_fn)
            for _ in range(3):
                idx = rng.choice(len(x_buf), size=min(32, len(x_buf)), replace=False)
                xb = mx.array(x_buf[idx])
                yb = mx.array(y_buf[idx])
                _, grads = grad_fn(model, xb, yb)
                opt.update(model, grads)
                mx.eval(model.parameters())
    elif op_name == "DOWNSCALE":
        # SHY downscale : multiply weights by 0.97
        for k, v in params.items():
            if "weight" in k or "bias" in k:
                params[k] = v * 0.97
        model.update(params)
    elif op_name == "RESTRUCTURE":
        # Sparsify : zero smallest 5% by magnitude
        for k, v in params.items():
            if "weight" in k:
                v_np = np.array(v)
                thresh = np.quantile(np.abs(v_np), 0.05)
                mask = np.abs(v_np) > thresh
                params[k] = mx.array(v_np * mask)
        model.update(params)
    elif op_name == "RECOMBINE":
        # Mix : tiny gaussian noise
        for k, v in params.items():
            if "weight" in k:
                noise = rng.normal(0, 0.005, size=v.shape).astype(np.float32)
                params[k] = v + mx.array(noise)
        model.update(params)


def train_with_order(order_perm, seed):
    np.random.seed(seed)
    mx.random.seed(seed)
    Xs, ys = make_cl_stream(seed=seed)
    n_tasks = len(Xs)
    dim_in = Xs[0].shape[1]
    n_classes = 2 * n_tasks
    model = MLP(dim_in, 32, n_classes)
    opt = optim.SGD(learning_rate=0.05)
    grad_fn = nn.value_and_grad(model, loss_fn)
    replay_buffer = (np.empty((0, dim_in), dtype=np.float32), np.empty(0, dtype=np.int32))
    for task_id in range(n_tasks):
        # Train on task
        for epoch in range(5):
            x, y = Xs[task_id], ys[task_id]
            xb = mx.array(x); yb = mx.array(y)
            _, grads = grad_fn(model, xb, yb)
            opt.update(model, grads)
            mx.eval(model.parameters())
        # Add to replay buffer (50 random samples per task)
        idx = np.random.choice(len(Xs[task_id]), size=50, replace=False)
        replay_buffer = (
            np.concatenate([replay_buffer[0], Xs[task_id][idx]]),
            np.concatenate([replay_buffer[1], ys[task_id][idx]]),
        )
        # Apply dream ops in specified order
        for i, op in enumerate(order_perm):
            apply_dream_op(model, op, replay_buffer, seed * 100 + task_id * 10 + i)
    # Eval on all tasks
    accs = []
    for t in range(n_tasks):
        logits = model(mx.array(Xs[t]))
        pred = mx.argmax(logits, axis=1)
        acc = float(mx.mean(pred == mx.array(ys[t])))
        accs.append(acc)
    return float(np.mean(accs)), accs


def main():
    OPS = ["REPLAY", "DOWNSCALE", "RESTRUCTURE", "RECOMBINE"]
    # Pick 6 permutations: canonical + 5 alternatives
    permutations = [
        ("REPLAY", "DOWNSCALE", "RESTRUCTURE", "RECOMBINE"),  # canonical
        ("DOWNSCALE", "REPLAY", "RESTRUCTURE", "RECOMBINE"),  # downscale-first
        ("REPLAY", "RESTRUCTURE", "DOWNSCALE", "RECOMBINE"),  # restructure-mid
        ("RECOMBINE", "REPLAY", "DOWNSCALE", "RESTRUCTURE"),  # recombine-first
        ("RESTRUCTURE", "RECOMBINE", "REPLAY", "DOWNSCALE"),  # reverse-canonical
        ("REPLAY", "RECOMBINE", "DOWNSCALE", "RESTRUCTURE"),  # replay-recombine
    ]
    seeds = [42, 43, 44, 45, 46]
    results = {}
    print(f"Robertson sequential-ordering test: {len(permutations)} perms x {len(seeds)} seeds = {len(permutations)*len(seeds)} cells")
    for perm in permutations:
        key = "→".join([op[:3] for op in perm])
        accs = []
        for seed in seeds:
            mean_acc, _ = train_with_order(list(perm), seed)
            accs.append(mean_acc)
        results[key] = {
            "perm": list(perm),
            "accs": accs,
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
        }
        print(f"  {key}: mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")
    # Compute pairwise Hedges' g vs canonical
    canonical = list(results.values())[0]["accs"]
    print(f"\nPairwise Hedges' g vs canonical (REP→DOW→RES→REC):")
    for key, r in list(results.items())[1:]:
        accs = r["accs"]
        pooled_sd = np.sqrt((np.var(canonical, ddof=1) + np.var(accs, ddof=1)) / 2)
        if pooled_sd > 0:
            g = (np.mean(accs) - np.mean(canonical)) / pooled_sd
            # Hedges' correction
            n1, n2 = len(canonical), len(accs)
            j = 1 - 3 / (4*(n1+n2) - 9)
            g *= j
        else:
            g = 0.0
        r["hedges_g_vs_canonical"] = float(g)
        print(f"  {key}: g={g:+.3f}")
    # Verdict
    max_g = max(abs(r.get("hedges_g_vs_canonical", 0)) for r in list(results.values())[1:])
    if max_g < 0.2:
        verdict = "H_RO-A: permutation effect SMALL (max |g| < 0.2) — order does not materially affect retention"
    elif max_g >= 0.4:
        verdict = f"H_RO-B: meaningful order effect detected (max |g| = {max_g:.3f})"
    else:
        verdict = f"H_RO-C: ambiguous (max |g| = {max_g:.3f}, between 0.2 and 0.4)"
    print(f"\nVERDICT: {verdict}")
    Path("/tmp/robertson_sequential_result.json").write_text(json.dumps({
        "permutations": results,
        "max_abs_hedges_g": float(max_g),
        "verdict": verdict,
    }, indent=2))
    print("Saved /tmp/robertson_sequential_result.json")


if __name__ == "__main__":
    main()
