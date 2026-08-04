"""Model architecture configuration constants.

These constants define model architecture variants and distance metrics
used by specific classifier implementations.
"""

# DeepSets architecture variants
DEEPSETS_ARCHITECTURE_BASE = "base"
"""Base Deep Sets architecture: simple encoder + pooling + head."""

DEEPSETS_ARCHITECTURE_PLUSPLUS = "plusplus"
"""Deep Sets++ architecture: residual connections, layer norm, gated head."""

DEEPSETS_ARCHITECTURES = (
  DEEPSETS_ARCHITECTURE_BASE,
  DEEPSETS_ARCHITECTURE_PLUSPLUS,
)
"""Valid Deep Sets architecture variants."""


# Prototype classifier distance metrics
PROTOTYPE_DISTANCE_EUCLIDEAN = "euclidean"
"""Euclidean distance for prototype-based classification.

Currently the only supported distance metric for PrototypeClassifier.
"""
