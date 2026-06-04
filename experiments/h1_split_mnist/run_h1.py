"""
H1 empirical validation — Split-MNIST class-incremental, 5 tasks, 3 seeds, 3 conditions.

The 3 conditions map to dreamOfkiki framework profiles :
  baseline : Naive finetune (no replay, no regulariser) — represents the "no-dream baseline"
  P_min    : Replay (buffer replay only)              — replay + downscale (weight decay only)
  P_equ    : Replay + EWC                             — replay + structural regularisation (EWC as Friston-FEP-style proxy)

All code below uses Avalanche library — we write no new algorithm.

Output : results_h1.csv with columns
  seed, condition, task_id, accuracy, forgetting
"""

import argparse
import csv
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.classic import SplitFMNIST
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, Replay


CONDITIONS = {
    "baseline": "Naive",                                 # no mitigation
    "P_min":    "Replay(mem_size=500)",                  # replay only
    "P_equ":    "Replay(mem_size=500) + EWC(λ=0.4)",     # replay + structural regulariser
}


def build_strategy(name, model, optimiser, loss_fn, plugin, device):
    if name == "baseline":
        return Naive(
            model=model, optimizer=optimiser, criterion=loss_fn,
            train_mb_size=64, train_epochs=5, eval_mb_size=128,
            device=device, evaluator=plugin,
        )
    if name == "P_min":
        return Replay(
            model=model, optimizer=optimiser, criterion=loss_fn,
            mem_size=500,
            train_mb_size=64, train_epochs=5, eval_mb_size=128,
            device=device, evaluator=plugin,
        )
    if name == "P_equ":
        from avalanche.training.plugins import ReplayPlugin, EWCPlugin
        plugins = [ReplayPlugin(mem_size=500), EWCPlugin(ewc_lambda=0.4)]
        return Naive(
            model=model, optimizer=optimiser, criterion=loss_fn,
            train_mb_size=64, train_epochs=5, eval_mb_size=128,
            device=device, evaluator=plugin, plugins=plugins,
        )
    raise ValueError(f"unknown condition {name}")


def run_one(seed: int, condition: str, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # SplitMNIST : 5 tasks, 2 classes each (standard class-incremental setup)
    benchmark = SplitFMNIST(n_experiences=5, seed=seed, return_task_id=False, shuffle=True)

    model = SimpleMLP(num_classes=10)
    optimiser = SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = CrossEntropyLoss()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger()],
    )

    strategy = build_strategy(condition, model, optimiser, loss_fn, eval_plugin, device)

    # Train sequentially on each experience, eval on all seen experiences after each
    for exp in benchmark.train_stream:
        strategy.train(exp)
        strategy.eval(benchmark.test_stream)

    # Pull final results from the evaluator
    final_results = strategy.evaluator.get_all_metrics()
    return final_results


def extract_metrics(all_results, seed, condition):
    """Extract per-task accuracy and forgetting from avalanche's metrics dict."""
    rows = []
    # Accuracy per experience (final state after training on all tasks)
    acc_stream_key = [k for k in all_results if "Top1_Acc_Exp" in k and "/eval_phase" in k]
    fgt_stream_key = [k for k in all_results if "Experience_Forgetting" in k and "/eval_phase" in k]

    # Last measurement per experience = final task accuracy / forgetting
    for i in range(5):
        acc_keys = [k for k in acc_stream_key if f"Exp{i:03d}" in k]
        fgt_keys = [k for k in fgt_stream_key if f"Exp{i:03d}" in k]
        final_acc = all_results[acc_keys[0]][1][-1] if acc_keys and all_results[acc_keys[0]][1] else None
        final_fgt = all_results[fgt_keys[0]][1][-1] if fgt_keys and all_results[fgt_keys[0]][1] else None
        rows.append({
            "seed": seed,
            "condition": condition,
            "task_id": i,
            "accuracy": float(final_acc) if final_acc is not None else None,
            "forgetting": float(final_fgt) if final_fgt is not None else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--out", default="results_h1.csv")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Conditions: {args.conditions}")
    print(f"Seeds: {args.seeds}")
    print(f"Total runs: {len(args.seeds) * len(args.conditions)}")
    print()

    all_rows = []
    t0 = time.time()
    for condition in args.conditions:
        for seed in args.seeds:
            run_start = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] >>> {condition} seed={seed} ...")
            try:
                results = run_one(seed, condition, args.device)
                rows = extract_metrics(results, seed, condition)
                all_rows.extend(rows)
                runtime = time.time() - run_start
                print(f"    done in {runtime:.1f}s — {len(rows)} metric rows")
                # Save intermediate after each run
                save_csv(all_rows, args.out)
            except Exception as e:
                print(f"    FAIL : {e}")
                import traceback
                traceback.print_exc()
    t_total = time.time() - t0
    print(f"\nTotal runtime : {t_total:.1f}s = {t_total/60:.1f} min")
    save_csv(all_rows, args.out)
    print(f"\nSaved : {args.out}")


def save_csv(rows, path):
    if not rows:
        return
    fieldnames = ["seed", "condition", "task_id", "accuracy", "forgetting"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
