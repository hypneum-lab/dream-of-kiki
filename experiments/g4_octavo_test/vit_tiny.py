"""ViT-tiny MLX substrate for G4-octavo Step 1 — NHWC layout.

Architecture (NHWC for the Conv2d patch embedding ; ``(N, S, D)``
for the transformer trunk where ``S = 1 + (H/p) * (W/p) = 257``
and ``D = dim = 192``) ::

    (N, 64, 64, 3)
        -> PatchEmbedding (Conv2d(3, 192, kernel=4, stride=4))
                                                  -> (N, 16, 16, 192)
        -> flatten + transpose                    -> (N, 256, 192)
        -> prepend class token                    -> (N, 257, 192)
        -> + positional embedding (1, 257, 192)   -> (N, 257, 192)
        -> 4 x TransformerBlock (LN -> MHA -> LN -> MLP)
                                                  -> (N, 257, 192)
        -> final LayerNorm                        -> (N, 257, 192)
        -> take class token x[:, 0]               -> (N, 192)
        -> Linear(192, n_classes)                 -> (N, n_classes)

Op site mapping (mirrors :mod:`experiments.g4_septimo_test.medium_cnn`) :

- **REPLAY** — full-model SGD on a batch from the beta buffer
  (records carry NHWC arrays of shape ``(64, 64, 3)``).
- **DOWNSCALE** — multiply every weight + bias of {patch_embed,
  blocks[*], norm, head} by ``factor``. Bound ``(0, 1]`` identical
  to the medium CNN.
- **RESTRUCTURE** — perturb the *last* TransformerBlock's MLP
  ``fc1`` weight only (analogue of the medium-CNN ``conv3``
  middle/late-layer perturbation ; preserves earlier blocks +
  patch embedding + the head classifier).
- **RECOMBINE** — synthetic latents (dim = ``dim``) per the active
  strategy ∈ {mog, none} ; one CE-loss SGD step on ``head`` only
  (the final ``Linear(192 -> n_classes)`` classifier).

The ViT-tiny dropout is 0.0 per pre-reg §5 ; this module deliberately
exposes no dropout knobs.

DR-0 accountability is provided by the dream-episode wrapper —
this module exposes only the model + train/eval primitives.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-octavo-pilot.md sec 2 (H7-A), sec 5
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map


class _PatchEmbed(nn.Module):
    """Patch embedding via ``Conv2d(3, dim, kernel=patch, stride=patch)``.

    Input is NHWC ``(N, 64, 64, 3)`` ; output is ``(N, S, D)`` with
    ``S = (64 / patch) ** 2`` and ``D = dim``.
    """

    def __init__(self, *, in_channels: int, dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch, stride=patch
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.proj(x)  # (N, H/p, W/p, D)
        n, hh, ww, d = h.shape
        return mx.reshape(h, (n, hh * ww, d))


class _MLPBlock(nn.Module):
    """Standard ViT MLP : ``Linear -> GELU -> Linear``.

    Dropout is 0.0 per pre-reg §5 ; not exposed.
    """

    def __init__(self, *, dim: int, mlp_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block : ``LN -> MHA -> + ; LN -> MLP -> +``.

    Mirrors the standard ViT block (Dosovitskiy 2021). MHA is via
    :class:`mlx.nn.MultiHeadAttention` with ``num_heads = heads`` and
    ``dims = dim``. Dropout is 0.0 per pre-reg §5.
    """

    def __init__(
        self, *, dim: int, heads: int, mlp_dim: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiHeadAttention(dims=dim, num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLPBlock(dim=dim, mlp_dim=mlp_dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm1(x)
        x = x + self.attn(h, h, h)
        x = x + self.mlp(self.norm2(x))
        return x


class _ViTTinyModule(nn.Module):
    """Internal :class:`mlx.nn.Module` wiring the ViT-tiny graph.

    Owns the parameter tree for ``mx.eval(self.parameters())`` and
    ``nn.value_and_grad`` ; the public dataclass :class:`G4ViTTiny`
    holds this module and exposes the substrate API expected by the
    dream-episode wrapper (see G4MediumCNN for the contract).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        patch: int,
        n_patches: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(
            in_channels=in_channels, dim=dim, patch=patch
        )
        self.cls_token = mx.zeros((1, 1, dim))
        self.pos_embed = mx.random.normal(
            shape=(1, n_patches + 1, dim), scale=0.02
        )
        self.blocks = [
            _TransformerBlock(dim=dim, heads=heads, mlp_dim=mlp_dim)
            for _ in range(depth)
        ]
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.patch_embed(x)  # (N, n_patches, D)
        n = h.shape[0]
        cls = mx.broadcast_to(self.cls_token, (n, 1, h.shape[2]))
        h = mx.concatenate([cls, h], axis=1)  # (N, n_patches+1, D)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h[:, 0])

    def encode_cls(self, x: mx.array) -> mx.array:
        """Return the post-final-LayerNorm class token ``(N, dim)``.

        This is the RECOMBINE Gaussian-MoG sampling site — the
        analogue of the medium-CNN post-``fc1`` ReLU activation.
        """
        h = self.patch_embed(x)
        n = h.shape[0]
        cls = mx.broadcast_to(self.cls_token, (n, 1, h.shape[2]))
        h = mx.concatenate([cls, h], axis=1)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h[:, 0]


@dataclass
class G4ViTTiny:
    """ViT-tiny classifier for Split-Tiny-ImageNet 20-class tasks.

    Parameters per pre-reg §5 : ``patch = 4``, ``dim = 192``,
    ``depth = 4``, ``heads = 3``, ``mlp_dim = 384``, ``dropout = 0.0``,
    ``n_classes = 20``. With 64×64 RGB inputs the patch grid is
    ``16 × 16 = 256`` patches and the class-token-prepended sequence
    length is 257. Deterministic under a fixed ``seed`` via
    ``mx.random.seed`` at construction.
    """

    dim: int = 192
    depth: int = 4
    heads: int = 3
    mlp_dim: int = 384
    patch: int = 4
    image_size: int = 64
    in_channels: int = 3
    n_classes: int = 20
    seed: int = 0
    _model: _ViTTinyModule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.image_size % self.patch != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"patch {self.patch}"
            )
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        n_patches = (self.image_size // self.patch) ** 2
        self._model = _ViTTinyModule(
            in_channels=self.in_channels,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
            patch=self.patch,
            n_patches=n_patches,
            n_classes=self.n_classes,
        )
        mx.eval(self._model.parameters())

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass returning raw logits ``(N, n_classes)``.

        Convenience for smoke tests + downstream wrappers ; the
        body is identical to ``self._model(x)``.
        """
        return self._model(x)

    @property
    def _head(self) -> nn.Linear:
        """Final classifier ``Linear(dim, n_classes)`` — RECOMBINE site."""
        return self._model.head

    @property
    def _last_block(self) -> _TransformerBlock:
        """Last transformer block — RESTRUCTURE perturbation site."""
        return self._model.blocks[-1]

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return post-final-LayerNorm class-token activations.

        Shape ``(N, dim)``. These are the activations *after the
        final ``LayerNorm``* on the class token — the RECOMBINE
        Gaussian-MoG sampling site for the ViT-tiny substrate
        (analogue of the medium-CNN post-``fc1`` ReLU latent).
        """
        h = self._model.encode_cls(mx.array(x))
        mx.eval(h)
        return np.asarray(h)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    def train_task(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Train one task on NHWC inputs.

        Accepts ``(x_train, y_train)`` arrays directly (ViT consumes
        NHWC via the patch-embedding Conv2d ; no flat-dict wrapper).
        """
        x = mx.array(x_train)
        y = mx.array(y_train)
        n = x.shape[0]
        opt = optim.SGD(learning_rate=lr)
        rng = np.random.default_rng(self.seed)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x[mx.array(idx)]
                yb = y[mx.array(idx)]
                _loss, grads = loss_and_grad(self._model, xb, yb)
                opt.update(self._model, grads)
                mx.eval(self._model.parameters(), opt.state)

    def restructure_step(self, *, factor: float, seed: int) -> None:
        """Add ``factor * sigma * N(0, 1)`` to the last block's MLP fc1.

        ``sigma`` is the per-tensor std of
        ``self._last_block.mlp.fc1.weight`` at call time. ``factor=0``
        is a no-op ; ``factor < 0`` raises. The site is intentionally
        late in the trunk (mirrors the G4MediumCNN ``conv3`` middle/
        late perturbation) and leaves the patch embedding, earlier
        blocks, and the classifier head untouched.
        """
        if factor < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if factor == 0.0:
            return
        target = self._last_block.mlp.fc1
        w = np.asarray(target.weight)
        sigma = float(w.std()) if w.size > 0 else 0.0
        if sigma == 0.0:
            return
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=w.shape).astype(np.float32)
        new_w = w + factor * sigma * noise
        target.weight = mx.array(new_w)
        mx.eval(target.weight)

    def downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Bounds : ``factor`` must lie in ``(0, 1]``. Touches the
        patch-embedding Conv2d, every transformer block (LayerNorms,
        MHA projections, MLP), the final LayerNorm and the classifier
        head — i.e. the full parameter tree returned by
        ``self._model.parameters()``.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        def _scale(value: mx.array) -> mx.array:
            return value * factor

        new_params = tree_map(_scale, self._model.parameters())
        self._model.update(new_params)
        mx.eval(self._model.parameters())

    def replay_optimizer_step(
        self,
        records: list[dict[str, list[float] | int]],
        *,
        lr: float,
        n_steps: int,
    ) -> None:
        """Replay-buffer SGD pass on the full ViT.

        ``records`` carry ``"x"`` as a nested NHWC structure of
        shape ``(64, 64, 3)``. Flat-shaped (12288-element) inputs
        are reshaped to NHWC for backwards compatibility with the
        small/medium-CNN code path.
        """
        if not records:
            return
        x_np = np.asarray([r["x"] for r in records], dtype=np.float32)
        if x_np.ndim == 2:
            # Flat 12288-d records reshape into NHWC for ViT consumption.
            x_np = x_np.reshape(-1, 64, 64, 3)
        y_np = np.asarray([r["y"] for r in records], dtype=np.int64)
        x = mx.array(x_np)
        y = mx.array(y_np)
        opt = optim.SGD(learning_rate=lr)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(n_steps):
            _loss, grads = loss_and_grad(self._model, x, y)
            opt.update(self._model, grads)
            mx.eval(self._model.parameters(), opt.state)

    def recombine_step(
        self,
        *,
        latents: list[tuple[list[float], int]],
        n_synthetic: int,
        lr: float,
        seed: int,
    ) -> None:
        """Sample ``n_synthetic`` synthetic latents from a per-class
        Gaussian-MoG and run one CE-loss SGD pass through ``head`` only.

        Empty / single-class ``latents`` -> no-op (S1-trivial).
        """
        if not latents:
            return
        classes = sorted({lbl for _, lbl in latents})
        if len(classes) < 2:
            return

        rng = np.random.default_rng(seed)
        components: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in classes:
            arr = np.asarray(
                [lat for lat, lbl in latents if lbl == c],
                dtype=np.float32,
            )
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-6
            components[c] = (mean, std)

        per_class = max(1, n_synthetic // len(classes))
        synth_x: list[np.ndarray] = []
        synth_y: list[int] = []
        for c in classes:
            mean, std = components[c]
            for _ in range(per_class):
                synth_x.append(
                    mean + std * rng.standard_normal(size=mean.shape).astype(
                        np.float32
                    )
                )
                synth_y.append(c)

        # Synthetic latents have dim = self.dim ; feed head directly.
        x_lat = mx.array(np.stack(synth_x).astype(np.float32))
        y = mx.array(np.asarray(synth_y, dtype=np.int32))

        opt = optim.SGD(learning_rate=lr)
        head = self._head

        def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(head, loss_fn)
        _loss, grads = loss_and_grad(head, x_lat, y)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state)
