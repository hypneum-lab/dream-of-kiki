"""G4-octavo pilot — H7-A Tiny-ImageNet ViT-tiny RECOMBINE-strategy placebo.

Fires the small-transformer follow-up to G4-septimo (H6-C universality
**confirmed** at the four-benchmark × four-CNN-or-MLP scope ceiling,
commit ``c8dd268``) by swapping the ``G4MediumCNN`` substrate for a
``G4ViTTiny`` Vision Transformer (patch=4, dim=192, depth=4, heads=3,
mlp_dim=384, n_classes=20 per task, 64×64 RGB input ; ~1.8 M params).
The Tiny-ImageNet loader is re-used verbatim from G4-septimo to
preserve cell-level R1 parity ; only the architecture changes.

Pre-registration : ``docs/osf-prereg-g4-octavo-pilot.md``.
"""
