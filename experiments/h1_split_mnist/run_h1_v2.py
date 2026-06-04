"""H1 empirical validation v2 — PermutedMNIST, CPU, proper per-experience extraction."""
import argparse
import csv
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.evaluation.metrics import accuracy_metrics, forgetting_metrics
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin, ReplayPlugin, EWCPlugin
from avalanche.training.supervised import Naive

CONDITIONS = ["baseline", "P_min", "P_equ"]

def build_plugins(cond):
    if cond == "baseline": return []
    if cond == "P_min":    return [ReplayPlugin(mem_size=500)]
    if cond == "P_equ":    return [ReplayPlugin(mem_size=500), EWCPlugin(ewc_lambda=0.4)]

def run_one(seed, cond, n_experiences=5, epochs=3):
    torch.manual_seed(seed)
    bench = PermutedMNIST(n_experiences=n_experiences, seed=seed)
    model = SimpleMLP(num_classes=10)
    opt = SGD(model.parameters(), lr=0.01, momentum=0.9)
    plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[],
    )
    strat = Naive(model=model, optimizer=opt, criterion=CrossEntropyLoss(),
                  train_mb_size=64, train_epochs=epochs, eval_mb_size=128,
                  device="cpu", evaluator=plugin, plugins=build_plugins(cond))
    # Eval before training = baseline random accuracy
    strat.eval(bench.test_stream)
    # Train + eval after each exp
    for exp in bench.train_stream:
        strat.train(exp)
        strat.eval(bench.test_stream)
    return strat.evaluator.get_all_metrics()

def extract(metrics, n_exp=5):
    """For each experience i, return (peak_accuracy, final_accuracy, forgetting)."""
    rows = []
    for i in range(n_exp):
        key_acc = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{i:03d}"
        key_fgt = f"ExperienceForgetting/eval_phase/test_stream/Task000/Exp{i:03d}"
        if key_acc in metrics:
            _, accs = metrics[key_acc]
            peak = max(accs)
            final = accs[-1]
            forgetting = peak - final
        else:
            peak = final = forgetting = None
        rows.append({"exp": i, "peak_acc": peak, "final_acc": final, "forgetting": forgetting})
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--n_experiences", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default="results_h1_v2.csv")
    args = parser.parse_args()

    print(f"Device: CPU | Benchmark: PermutedMNIST-{args.n_experiences} | Epochs: {args.epochs}")
    print(f"Conditions: {args.conditions} | Seeds: {args.seeds}")
    n_runs = len(args.conditions) * len(args.seeds)
    print(f"Total runs: {n_runs}\n")

    all_rows = []
    t0 = time.time()
    for condition in args.conditions:
        for seed in args.seeds:
            t1 = time.time()
            print(f"[{time.strftime('%H:%M:%S')}] >>> {condition} seed={seed}")
            metrics = run_one(seed, condition, args.n_experiences, args.epochs)
            rows = extract(metrics, args.n_experiences)
            for r in rows:
                all_rows.append({"seed": seed, "condition": condition, **r})
            dt = time.time() - t1
            mean_fgt = sum(r['forgetting'] for r in rows if r['forgetting'] is not None) / len(rows)
            mean_final = sum(r['final_acc'] for r in rows if r['final_acc'] is not None) / len(rows)
            print(f"    runtime {dt:.1f}s | mean_final_acc {mean_final:.4f} | mean_forgetting {mean_fgt:.4f}")
            save_csv(all_rows, args.out)
    print(f"\nTotal runtime {time.time()-t0:.1f}s = {(time.time()-t0)/60:.1f} min")
    save_csv(all_rows, args.out)

def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed","condition","exp","peak_acc","final_acc","forgetting"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

if __name__ == "__main__":
    main()
