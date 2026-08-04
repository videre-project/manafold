"""Pooling mode constants for classifier architectures.

These constants define how token embeddings are aggregated into a single
deck-level representation. Used by Deep Sets, Set Transformer, and
pooled linear classifiers.
"""

# Standard pooling modes for Deep Sets and Pooled Linear classifiers
POOLING_SUM = "sum"
"""Sum pooling: aggregate token embeddings by summation."""

POOLING_MEAN = "mean"
"""Mean pooling: aggregate token embeddings by averaging."""

POOLING_QUANTITY_WEIGHTED = "quantity-weighted"
"""Quantity-weighted pooling: weight each token by its card quantity."""

POOLING_MODES = (POOLING_SUM, POOLING_MEAN, POOLING_QUANTITY_WEIGHTED)
"""Valid pooling modes for Deep Sets and Pooled Linear classifiers."""


# Set Transformer specific pooling modes
SET_TRANSFORMER_POOLING_PMA = "pma"
"""Pooling by Multihead Attention (PMA): learnable attention-based pooling."""

SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED = "quantity-weighted"
"""Quantity-weighted pooling for Set Transformer: weight tokens by quantity."""

SET_TRANSFORMER_POOLING_HYPERGEOMETRIC = "hypergeometric"
"""Hypergeometric pooling: sample tokens without replacement (simulates opening hand)."""

SET_TRANSFORMER_POOLING_MODES = (
  SET_TRANSFORMER_POOLING_PMA,
  SET_TRANSFORMER_POOLING_QUANTITY_WEIGHTED,
  SET_TRANSFORMER_POOLING_HYPERGEOMETRIC,
)
"""Valid pooling modes for Set Transformer classifiers."""
