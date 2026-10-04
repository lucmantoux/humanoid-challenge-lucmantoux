"""Put src/ on sys.path so the scripts run without relying on the editable install.

`uv sync` installs palm_prior editable, which works by dropping a .pth file into
site-packages. Python ignores a .pth file that the filesystem marks hidden, and on macOS
iCloud marks everything under a synced .venv hidden — so the install silently stops
working. One import removes that whole class of failure, on every platform.

Import it first, before anything from palm_prior.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
