"""Partial observation training configuration constants.

These constants control the partial observation training regime where
the model sees only a subset of card identities during training,
simulating incomplete deck information.
"""

# Default identity counts for partial observation training
DEFAULT_PARTIAL_OBSERVATION_IDENTITY_COUNTS = (5, 10, 20)
"""Number of card identities to observe in partial training views.

Each training example gets a random subset of this many unique card
identities. The model must predict the full archetype from this
partial observation.
"""

# Loss weights for partial observation training
DEFAULT_PARTIAL_CLASSIFICATION_WEIGHT = 0.5
"""Weight for the classification loss on partial observations."""

DEFAULT_PARTIAL_CONSISTENCY_WEIGHT = 0.5
"""Weight for the consistency loss between partial and full predictions."""

DEFAULT_PARTIAL_MIN_COVERAGE_WEIGHT = 0.25
"""Weight for the minimum coverage regularization term."""

DEFAULT_PARTIAL_LATENT_WEIGHT = 0.5
"""Weight for the latent space prediction loss."""

DEFAULT_PARTIAL_TEACHER_DECAY = 0.99
"""EMA decay rate for the teacher model in partial observation training."""


# Corruption policies for generating partial observations
PARTIAL_CORRUPTION_FIXED = "fixed_identity_counts"
"""Fixed corruption: observe exactly N random card identities per example."""

PARTIAL_CORRUPTION_MIXTURE = "regular_extreme_evidence"
"""Mixture corruption: alternate between regular and extreme evidence regimes."""

PARTIAL_CORRUPTION_POLICIES = (
  PARTIAL_CORRUPTION_FIXED,
  PARTIAL_CORRUPTION_MIXTURE,
)
"""Valid partial observation corruption policies."""
