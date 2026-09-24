"""Command-line entry points.

The CLIs live here rather than in ``scripts/`` so that each one is importable,
testable and installable as a console script (``gtsrb-train``, ``gtsrb-evaluate``,
...). ``scripts/`` is kept for one-off utilities that are genuinely not part of the
package's interface, such as dataset preparation.
"""

from __future__ import annotations

__all__: list[str] = []
