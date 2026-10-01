"""Locate the nmppfc package, which supplies the vehicle model, path and controller.

This repository does not re-implement the vehicle or the controller. They already exist:

    https://github.com/nahueldeniz/nmppfc

which is the Python + CasADi implementation of the path-following framework of
Deniz and Auat Cheein, IEEE T-ASE 22:7018-7032, 2025. ``requirements.txt`` says how to
obtain it.

Failing an installed copy, two fallbacks are tried, in this order: ``$NMPPFC_PATH``, and the
sibling checkout ``../../02_nmppfc/code`` that exists when this repository sits inside the
paper-videos tree.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_SIBLINGS = [Path(p).expanduser() for p in (os.environ.get("NMPPFC_PATH"),) if p]
_SIBLINGS += [Path(__file__).resolve().parents[3] / "02_nmppfc" / "code"]

try:  # installed, or already on the path
    import nmppfc  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - development convenience
    for _p in _SIBLINGS:
        if (_p / "nmppfc" / "__init__.py").exists():
            sys.path.insert(0, str(_p))
            break
    else:
        raise ModuleNotFoundError(
            "nmppfc not found. Install it (see requirements.txt) or set NMPPFC_PATH to a "
            "checkout of https://github.com/nahueldeniz/nmppfc")
    import nmppfc  # noqa: F401

from nmppfc.model import NTrailer, VehicleGeometry  # noqa: E402

# This repository needs the redundant-coordinate helpers that were added to nmppfc so its
# controllers accept the full state vector q the estimators produce. An older checkout does
# not have them, and without this check the first failure is an obscure AttributeError deep
# inside the model. Fail here instead, saying what to do about it.
_REQUIRED = ("nq", "q_layout", "q_from_x", "x_from_q", "as_minimal", "chain_residual")
_missing = [a for a in _REQUIRED if not hasattr(NTrailer, a)]
if _missing:
    raise ImportError(
        "the nmppfc checkout at "
        + str(next((p for p in sys.path if (Path(p) / "nmppfc" / "model.py").exists()), "?"))
        + " predates the redundant-coordinate support this package relies on (missing: "
        + ", ".join(_missing) + "). Update it:\n"
        "    cd <nmppfc checkout> && git pull\n"
        "or clone a current copy from https://github.com/nahueldeniz/nmppfc and point\n"
        "NMPPFC_PATH at it. See NOTES.md, section 3.")

__all__ = ["nmppfc", "NTrailer", "VehicleGeometry"]
