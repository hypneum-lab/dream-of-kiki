"""K2 real-substrate validation on SpikingKiki-V4 35B-A3B-V4.

Tests whether the K2 phase-coupling invariant [0.27, 0.39]
(eLife 2025 BF=58 anchor) holds on REAL spike trains generated
from the production SpikingKiki-V4 substrate, vs the synthetic
PAC_DEPTH=0.33 reference.

Workflow :
  1. Load lif_metadata.json (T=128, threshold=0.0625, tau=1.0)
  2. Sample N modules from KIKI-Mac_tunner SpikingKiki-V4
  3. For each module : load weights, simulate LIF, get spike rates
  4. For pairs of modules : compute phase coupling via mean-vector-length
  5. Aggregate MVL distribution, compare to K2 invariant

Multiprocessing : 24 P-cores in parallel (Studio M3 Ultra).
Compute estimate : ~5-10 min Studio.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path("/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4")

# Read lif_metadata.json
META = json.loads((ROOT / "lif_metadata.json").read_text())
T = META.get("T", 128)
THRESHOLD = META.get("threshold", 0.0625)
TAU = META.get("tau", 1.0)
print(f"LIF metadata: T={T}, threshold={THRESHOLD}, tau={TAU}")


def lif_simulate(weight: np.ndarray, n_input: int = 64, seed: int = 0) -> np.ndarray:
    """Simulate LIF neuron dynamics over T steps using weight tensor as input projection.

    Returns spike train shape (T, n_output) where n_output = weight.shape[0].
    """
    rng = np.random.default_rng(seed)
    # Use first 2D slice of weight if higher-dim
    W = weight.reshape(weight.shape[0], -1) if weight.ndim > 1 else weight.reshape(-1, 1)
    n_out, n_w = W.shape
    # Generate Poisson-like input rates
    inputs = rng.poisson(0.5, size=(T, n_input)).astype(np.float32)
    # Project via random subset of W columns (first n_input)
    if n_w >= n_input:
        proj = W[:, :n_input]
    else:
        proj = np.tile(W, (1, (n_input // n_w) + 1))[:, :n_input]
    drive = inputs @ proj.T  # shape (T, n_out)
    # LIF dynamics
    v = np.zeros(n_out, dtype=np.float32)
    spikes = np.zeros((T, n_out), dtype=np.float32)
    for t in range(T):
        v = (1 - 1.0/TAU) * v + drive[t] / TAU
        fire = v > THRESHOLD
        spikes[t] = fire.astype(np.float32)
        v = np.where(fire, 0.0, v)
    return spikes


def measure_mvl(spike_train_a: np.ndarray, spike_train_b: np.ndarray) -> float:
    """Mean vector length of phase coupling between two spike rate signals."""
    # Convert spikes to instantaneous rate via boxcar smoothing
    rate_a = np.convolve(spike_train_a.mean(axis=1), np.ones(8)/8, mode='same')
    rate_b = np.convolve(spike_train_b.mean(axis=1), np.ones(8)/8, mode='same')
    # Hilbert-like phase: use FFT phase
    phase_a = np.angle(np.fft.fft(rate_a - rate_a.mean()))
    phase_b = np.angle(np.fft.fft(rate_b - rate_b.mean()))
    # MVL = magnitude of mean complex vector at phase difference
    diff = phase_a - phase_b
    mvl = np.abs(np.mean(np.exp(1j * diff)))
    return float(mvl)


def process_module(args):
    npz_path, seed = args
    try:
        data = np.load(npz_path)
        # Get first array key
        key = list(data.keys())[0]
        weight = data[key]
        spikes = lif_simulate(weight, seed=seed)
        return (npz_path.name, spikes)
    except Exception as e:
        return (npz_path.name, None)


def main():
    # Sample 24 modules
    npz_files = sorted(ROOT.glob("model_layers_*.npz"))
    print(f"Found {len(npz_files)} .npz modules; sampling 24 for K2 test")
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(npz_files), size=min(24, len(npz_files)), replace=False)
    samples = [(npz_files[i], int(i)) for i in sample_indices]

    print(f"Spawning {len(samples)} parallel LIF simulations on Studio CPU...")
    with Pool(processes=24) as pool:
        results = pool.map(process_module, samples)

    spike_trains = {name: spikes for name, spikes in results if spikes is not None}
    print(f"Generated {len(spike_trains)} spike trains")

    # Compute MVL across all unique pairs
    names = list(spike_trains.keys())
    mvls = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            mvl = measure_mvl(spike_trains[names[i]], spike_trains[names[j]])
            mvls.append(mvl)
    mvls = np.array(mvls)

    print(f"\n=== K2 phase-coupling on real SpikingKiki-V4 spike trains ===")
    print(f"Pairs measured: {len(mvls)}")
    print(f"MVL mean: {mvls.mean():.4f}")
    print(f"MVL std:  {mvls.std():.4f}")
    print(f"MVL median: {np.median(mvls):.4f}")
    print(f"MVL 5-95 percentile: [{np.percentile(mvls, 5):.4f}, {np.percentile(mvls, 95):.4f}]")
    print(f"\nK2 invariant range: [0.27, 0.39] (eLife 2025 BF=58 anchor)")
    in_range = ((mvls >= 0.27) & (mvls <= 0.39)).mean()
    print(f"Fraction of pairs in K2 range: {in_range*100:.1f}%")
    if in_range > 0.5:
        print("VERDICT: K2 invariant EMPIRICALLY VALIDATED on real SpikingKiki-V4 spike trains")
    elif mvls.mean() > 0.39:
        print(f"VERDICT: real spikes show STRONGER coupling than synthetic anchor (mean {mvls.mean():.3f} > 0.39)")
    elif mvls.mean() < 0.27:
        print(f"VERDICT: real spikes show WEAKER coupling than synthetic anchor (mean {mvls.mean():.3f} < 0.27)")
    else:
        print(f"VERDICT: mixed (mean {mvls.mean():.3f} in range but spread wide)")

    # Save result
    result = {
        "n_pairs": int(len(mvls)),
        "mvl_mean": float(mvls.mean()),
        "mvl_std": float(mvls.std()),
        "mvl_median": float(np.median(mvls)),
        "mvl_p5": float(np.percentile(mvls, 5)),
        "mvl_p95": float(np.percentile(mvls, 95)),
        "k2_lower": 0.27,
        "k2_upper": 0.39,
        "fraction_in_range": float(in_range),
        "lif_metadata": {"T": T, "threshold": THRESHOLD, "tau": TAU},
        "n_modules_sampled": len(spike_trains),
    }
    Path("/tmp/k2_real_substrate_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to /tmp/k2_real_substrate_result.json")


if __name__ == "__main__":
    main()
