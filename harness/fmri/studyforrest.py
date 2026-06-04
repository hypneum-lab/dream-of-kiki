"""Studyforrest BOLD loader (cycle-3 C3.15 — Phase 2 track c).

Loads BOLD time-series from the ds000113 Studyforrest dataset
(CC0) and produces episode-wise (x, y, z, t) ndarrays with TR
+ HRF metadata and frame-aligned event indices ready for HMM
alignment (C3.16) + CCA (C3.17) downstream.

Dependency policy : ``nibabel`` is an **optional** dependency
(see ``[project.optional-dependencies]`` group ``fmri`` in
``pyproject.toml``). When not importable the loader falls back
to a ``.npy`` on-disk format so CI + local unit tests work
without the neuroimaging toolchain. A production run on the
full ds000113 tree requires ``pip install -e '.[fmri]'``.

Path resolution : ``STUDYFORREST_ROOT`` env var with default
``/mnt/models/studyforrest/ds000113`` (matches the kxkm-ai NAS
setup used by ``scripts/init_studyforrest_download.sh`` — see
commit ``7b79b9e``). When a ``root_path`` is passed explicitly
to the loader it wins.

References :
- docs/interfaces/fmri-schema.yaml (schema v0.7.0+PARTIAL)
- scripts/init_studyforrest_download.sh (download config)
- framework-C spec §6.2 (DR-3 Conformance Criterion), §3 track (c)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

# DualVer C-v0.7.0+PARTIAL — locked in step with the fmri-schema.yaml
# header (cycle-3 Phase 1 bump ; see CHANGELOG.md [C-v0.7.0+PARTIAL]
# and framework-C spec §12.3). Downstream harness code (C3.16 HMM,
# C3.17 CCA) cross-checks this constant against the schema version.
STUDYFORREST_LOADER_VERSION = "C-v0.7.0+PARTIAL"

# Default NAS root matches scripts/init_studyforrest_download.sh ;
# override with the STUDYFORREST_ROOT env var. The task spec's
# "/mnt/models/studyforrest/ds000113" path mirrors the production
# layout on kxkm-ai.
DEFAULT_STUDYFORREST_ROOT = "/mnt/models/studyforrest"

# Optional-dependency probe — nibabel is only needed for real
# NIfTI volumes. Unit tests + the numpy fixture path do not need
# it. The flag is exposed on the loader for introspection.
try:  # pragma: no cover - branch depends on env
    import nibabel  # noqa: F401
    _NIBABEL_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _NIBABEL_AVAILABLE = False


@dataclass
class BoldSeries:
    """Single BOLD run ready for downstream alignment.

    ``event_times`` are 1-D int indices into the BOLD frame axis
    (floor(onset_seconds / tr_seconds)) — the schema contract for
    HMM alignment (see fmri-schema.yaml §alignment).
    """

    subject_id: str
    task: str
    data: NDArray[np.floating]          # (x, y, z, t)
    tr_seconds: float
    event_times: NDArray[np.integer]    # (n_events,) frame indices
    run_label: str = ""


@dataclass
class StudyforrestLoader:
    """BOLD loader for ds000113 Studyforrest.

    Parameters
    ----------
    root_path
        Root directory. The dataset tree is expected under
        ``root_path / "ds000113"``. If not supplied, falls back
        to the ``STUDYFORREST_ROOT`` env var.
    tr_seconds
        Repetition time in seconds. Studyforrest movie runs are
        TR=2.0 s by default ; overridable per task.
    """

    root_path: Path = field(default=Path())
    tr_seconds: float = 2.0
    nibabel_available: bool = field(default=_NIBABEL_AVAILABLE, init=False)

    def __post_init__(self) -> None:
        if self.root_path in (Path(), Path("")):
            env = os.environ.get("STUDYFORREST_ROOT")
            self.root_path = Path(env) if env else Path(
                DEFAULT_STUDYFORREST_ROOT
            )
        else:
            self.root_path = Path(self.root_path)

        if not self.root_path.exists():
            raise FileNotFoundError(
                f"Studyforrest root does not exist : {self.root_path}"
            )

        # BIDS marker : dataset_description.json under ds000113/
        ds_dir = self.root_path / "ds000113"
        if not (ds_dir / "dataset_description.json").exists():
            raise ValueError(
                f"root_path={self.root_path!s} does not look like a "
                f"ds000113 tree (missing dataset_description.json)"
            )

    # ----- iteration -----

    def iter_bold_series(
        self, subject_id: str, task: str,
    ) -> Iterator[BoldSeries]:
        """Yield every BOLD run for (subject_id, task).

        Searches ``ds000113/<subject>/ses-*/func/*_<task>_bold.*``
        with a deterministic filename sort. Companion TSV file
        ``*_<task>_events.tsv`` provides the onsets → event_times.
        """
        subject_root = self.root_path / "ds000113" / subject_id
        if not subject_root.exists():
            return

        # Deterministic sort (critical for R1 contract)
        run_files = sorted(subject_root.glob(f"ses-*/func/*_{task}_*bold.*"))
        for bold_path in run_files:
            # Skip unsupported extensions silently
            data = self._load_volume(bold_path)
            if data is None:
                continue
            events_path = bold_path.parent / (
                bold_path.name.replace("_bold.npy", "_events.tsv")
                .replace("_bold.nii.gz", "_events.tsv")
                .replace("_bold.nii", "_events.tsv")
            )
            event_times = self._load_events(events_path)
            yield BoldSeries(
                subject_id=subject_id,
                task=task,
                data=data,
                tr_seconds=self.tr_seconds,
                event_times=event_times,
                run_label=bold_path.stem,
            )

    def _load_volume(self, path: Path) -> NDArray[np.floating] | None:
        """Load a 4-D BOLD volume from a .npy or .nii(.gz) file.

        ``.npy`` path is always available ; ``.nii(.gz)`` requires
        nibabel. Returns ``None`` for unrecognized suffixes so the
        caller can skip silently (used when a stub fixture writes
        a non-volume sibling file).
        """
        suffix = "".join(path.suffixes).lower()
        if suffix == ".npy":
            arr = np.load(path)
            return np.asarray(arr, dtype=np.float32)
        if suffix in (".nii", ".nii.gz") and self.nibabel_available:
            # pragma : the real nibabel branch is exercised only in
            # production environments with the fmri extras installed.
            import nibabel as nib  # pragma: no cover - env-gated

            img = nib.load(str(path))  # pragma: no cover - env-gated
            # get_fdata exists on SpatialImage subclasses; the
            # FileBasedImage return type of nib.load is too broad.
            return np.asarray(  # pragma: no cover - env-gated
                img.get_fdata(), dtype=np.float32,  # type: ignore[attr-defined]
            )
        return None

    def _load_events(self, path: Path) -> NDArray[np.integer]:
        """Parse a BIDS events.tsv → int frame indices.

        Format : header line + one event per row, tab-separated,
        first column = onset (seconds). Empty / missing file
        yields an empty array.
        """
        if not path.exists():
            return np.zeros(0, dtype=int)
        onsets: list[float] = []
        with path.open("r", encoding="utf-8") as fh:
            header = fh.readline()
            if not header:
                return np.zeros(0, dtype=int)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                cols = line.split("\t")
                try:
                    onsets.append(float(cols[0]))
                except ValueError:
                    continue
        if not onsets:
            return np.zeros(0, dtype=int)
        frames = np.floor(
            np.asarray(onsets, dtype=float) / self.tr_seconds,
        ).astype(int)
        return frames

    # ----- HRF helper -----

    @staticmethod
    def canonical_hrf(
        tr_seconds: float, duration_s: float = 32.0,
    ) -> NDArray[np.floating]:
        """Glover 1999 canonical double-gamma HRF.

        Standard parameters (matches nilearn.glm.first_level.glover_hrf) :
        peak ~ Gamma(shape=6, scale=1) ; undershoot ~
        Gamma(shape=16, scale=1) scaled by 1/6 ; superposition
        gives the characteristic positive peak at t ≈ 5-6 s and
        a small negative undershoot at t ≈ 12-15 s.

        Returns ``duration_s / tr_seconds`` samples, normalized so
        the max is 1.0.
        """
        if tr_seconds <= 0:
            raise ValueError(
                f"tr_seconds must be > 0, got {tr_seconds}"
            )
        # SciPy is already a project dependency (see pyproject.toml),
        # so this path is always available.
        from scipy.stats import gamma

        n_samples = int(round(duration_s / tr_seconds))
        t = np.arange(n_samples) * tr_seconds
        peak: NDArray[np.floating] = np.asarray(
            gamma.pdf(t, a=6.0, scale=1.0), dtype=np.float64,
        )
        undershoot: NDArray[np.floating] = np.asarray(
            gamma.pdf(t, a=16.0, scale=1.0), dtype=np.float64,
        )
        hrf = peak - undershoot / 6.0
        max_val = float(np.max(np.abs(hrf)))
        if max_val > 0.0:
            hrf = hrf / max_val
        return hrf.astype(np.float64)
