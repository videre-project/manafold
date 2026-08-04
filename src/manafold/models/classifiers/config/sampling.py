"""Training sampling and hypergeometric configuration constants.

These constants control how training examples are sampled and how
hypergeometric pooling simulates opening hand draws.
"""

# Hypergeometric pooling (simulates drawing an opening hand)
DEFAULT_HYPERGEOMETRIC_DRAW_COUNT = 7
"""Number of cards to draw in hypergeometric pooling.

Simulates a 7-card opening hand drawn without replacement from the deck.
Used by Set Transformer hypergeometric pooling mode.
"""


# Training sampling policies
TRAINING_SAMPLING_NATURAL = "natural"
"""Natural sampling: use the dataset's natural class distribution."""

TRAINING_SAMPLING_NATURAL_SQRT_BALANCED = "natural_sqrt_balanced"
"""Square-root balanced sampling: reweight classes by sqrt(count)."""

TRAINING_SAMPLING_POLICIES = (
  TRAINING_SAMPLING_NATURAL,
  TRAINING_SAMPLING_NATURAL_SQRT_BALANCED,
)
"""Valid training sampling policies."""


# Balanced sampling parameters
DEFAULT_BALANCED_SAMPLING_FRACTION = 0.5
"""Fraction of training steps to use balanced sampling (vs natural)."""

DEFAULT_BALANCED_SAMPLING_MAX_MULTIPLIER = 4.0
"""Maximum oversampling multiplier for minority classes in balanced sampling."""
