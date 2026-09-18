from __future__ import annotations

from pathlib import Path
import re

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvFileError(ValueError):
    pass


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a deliberately small dotenv subset without executing shell code."""
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvFileError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _KEY.fullmatch(key):
            raise EnvFileError(f"{path}:{number}: invalid environment variable name {key!r}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result
