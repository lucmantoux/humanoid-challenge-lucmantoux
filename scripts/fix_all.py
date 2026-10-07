"""Entry point that puts src/ on the path, then runs the v2 pipeline.

    uv run python scripts/fix_all.py
    uv run python scripts/fix_all.py --only e1
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from palm_prior.fix_all import main

if __name__ == "__main__":
    main()
