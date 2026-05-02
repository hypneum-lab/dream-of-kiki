"""Synthetic phase-coupling substrate (test-only fixture).

Documented as a synthetic substitute target for the K2 conformance
test. **Not** part of `kiki_oniric/` and **not** an empirical claim
substrate. Real substrates implementing
:class:`PhaseCouplingObservable` will replace it as they appear
(MLX kiki-oniric S18+, E-SNN thalamocortical S22+).

Construction
------------
SO carrier frequency f_so = 1.0 Hz, fast-spindle envelope modulated
by SO phase with depth 0.10 around mean 0.5, additive Gaussian
noise sigma = 0.05. Sampling rate fs = 256 Hz.

These values are calibrated so the mean-vector-length estimator
(see ``test_k2_coupling.test_k2_property_in_window``) yields a
coupling strength that falls inside the empirical 95 % CI
[0.27, 0.39] (eLife 2025 Bayesian meta-analysis,
``elife2025bayesian``).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SyntheticPhaseCouplingSubstrate:
    """Synthetic, deterministic phase-coupling source.

    Implements :class:`kiki_oniric.core.observables.PhaseCouplingObservable`
    structurally (no inheritance — Protocol is opt-in).
    """

    F_SO: float = 1.0       # Hz — slow oscillation
    FS: float = 256.0       # Hz — sampling rate
    PAC_DEPTH: float = 0.10  # modulation depth around mean amplitude
    AMP_MEAN: float = 0.5
    NOISE_SIGMA: float = 0.05

    def emit_phase_coupling_signal(
        self, n_samples: int, seed: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        rng = np.random.default_rng(seed)
        t = np.arange(n_samples, dtype=np.float64) / self.FS
        phase = 2.0 * np.pi * self.F_SO * t
        # Wrap phase to [-pi, pi] before downcast.
        phase_wrapped = np.arctan2(np.sin(phase), np.cos(phase))
        noise = rng.normal(0.0, self.NOISE_SIGMA, size=n_samples)
        amp = self.AMP_MEAN + self.PAC_DEPTH * np.cos(phase) + noise
        # Amplitude must be non-negative (it is an envelope magnitude).
        amp = np.clip(amp, a_min=0.0, a_max=None)
        return (
            phase_wrapped.astype(np.float32),
            amp.astype(np.float32),
            self.FS,
        )
