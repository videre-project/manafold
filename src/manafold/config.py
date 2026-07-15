from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatabaseSettings:
  dsn: str | None
  host: str | None
  port: str | None
  database: str | None
  user: str | None
  password: str | None

  @classmethod
  def from_environment(
    cls,
    repo_root: Path,
    env_file: Path | None = None,
  ) -> "DatabaseSettings":
    values: dict[str, str] = {}

    for candidate in [
      repo_root.parent / "mtgo-db" / ".env",
      repo_root.parent / "MTGOBot" / ".env",
      repo_root / ".env",
      env_file,
    ]:
      if candidate is not None:
        values.update(_read_env_file(candidate))

    values.update({key: value for key, value in os.environ.items() if value})

    dsn = values.get("MANAFOLD_DATABASE_URL") or values.get("DATABASE_URL")
    host = (
      values.get("PGHOST")
      or values.get("POSTGRES_HOST")
      or values.get("TAILSCALE_IP")
    )

    if values.get("PGPORT"):
      port = values["PGPORT"]
    elif host and host == values.get("TAILSCALE_IP"):
      port = values.get("TAILSCALE_POSTGRES_PORT", "5433")
    else:
      port = values.get("POSTGRES_PORT", "5432")

    return cls(
      dsn=dsn,
      host=host,
      port=port,
      database=values.get("PGDATABASE") or values.get("POSTGRES_DB"),
      user=values.get("PGUSER") or values.get("POSTGRES_USER"),
      password=values.get("PGPASSWORD") or values.get("POSTGRES_PASSWORD"),
    )

  def connect_kwargs(self) -> dict[str, str]:
    if self.dsn:
      return {"conninfo": self.dsn}

    missing = [
      name
      for name, value in {
        "host": self.host,
        "port": self.port,
        "dbname": self.database,
        "user": self.user,
        "password": self.password,
      }.items()
      if not value
    ]
    if missing:
      joined = ", ".join(missing)
      raise RuntimeError(f"Missing database connection settings: {joined}")

    return {
      "host": self.host or "",
      "port": self.port or "",
      "dbname": self.database or "",
      "user": self.user or "",
      "password": self.password or "",
    }


def repo_root() -> Path:
  if workspace := os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
    return Path(workspace)

  return Path(__file__).resolve().parents[2]


def _read_env_file(path: Path) -> dict[str, str]:
  if not path.exists():
    return {}

  values: dict[str, str] = {}
  for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
      continue

    key, value = stripped.split("=", 1)
    value = value.strip().strip("'").strip('"')
    values[key.strip()] = value

  return values
