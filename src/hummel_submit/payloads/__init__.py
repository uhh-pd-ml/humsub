"""Built-in payloads, selected with ``--payload builtin:<name>`` (copied into the submission like any payload)."""
from __future__ import annotations

from pathlib import Path

PREFIX = "builtin:"


def available() -> list[str]:
    return sorted(p.stem for p in Path(__file__).parent.glob("*.py") if not p.stem.startswith("_"))


def resolve(value: str) -> Path | None:
    """The file of a ``builtin:<name>`` payload; ``None`` if *value* is not a builtin reference."""
    if not value.startswith(PREFIX):
        return None
    name = value[len(PREFIX):]
    if name not in available():
        raise ValueError(f"unknown builtin payload {name!r}; available: {', '.join(available())}")
    return Path(__file__).parent / f"{name}.py"
