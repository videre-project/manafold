from __future__ import annotations

import math
from typing import Any


def summary_statistics(values: list[float | None]) -> dict[str, Any]:
  numeric_values = [value for value in values if value is not None]
  if not numeric_values:
    return {
      "count": 0,
      "mean": None,
      "std": None,
      "values": values,
    }

  mean = sum(numeric_values) / len(numeric_values)
  variance = (
    sum((value - mean) ** 2 for value in numeric_values) / len(numeric_values)
  )
  return {
    "count": len(numeric_values),
    "mean": mean,
    "std": math.sqrt(variance),
    "values": values,
  }
