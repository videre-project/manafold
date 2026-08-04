from __future__ import annotations

import os
from pathlib import Path
from typing import Final


PROJECT_ROOT: Final = Path(
  os.environ.get("BUILD_WORKSPACE_DIRECTORY", Path(__file__).resolve().parents[2])
)
