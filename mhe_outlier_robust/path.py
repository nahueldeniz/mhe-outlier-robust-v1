"""The agricultural path of Fig. 2, rebuilt from its geometry.

Fig. 2 of the paper shows five straight rows 20 m long and 1.5 m apart, joined at alternating
ends by bulb turns -- full circles the tractor drives round because a chained vehicle cannot
turn inside the row spacing. The circles are tangent to both rows, which fixes their centres:
a circle of radius ``r`` centred on the mid-line between two rows meets each row at a
horizontal offset ``sqrt(r^2 - (spacing/2)^2)`` from its centre, so the row must end there.

With ``r = 2`` m and a 1.5 m spacing the neighbouring circles at the same end overlap and
cross, which is what the published figure shows -- that crossing is the check that the
reconstruction is right, not a coincidence. This is a reconstruction from the figure, not the
paper's own path file.

The result is an ``nmppfc.path.Path`` parameterised by ARC LENGTH, so a commanded speed in
metres per second means the same thing everywhere on the curve regardless of how the rows and
the turns are sampled.
"""
from __future__ import annotations

import numpy as np

from ._nmppfc import nmppfc

ROW_LENGTH = 20.0
SPACING = 1.5
TURN_RADIUS = 2.0
N_ROWS = 5


def polyline(n_rows: int = N_ROWS, row_length: float = ROW_LENGTH, spacing: float = SPACING,
             turn_radius: float = TURN_RADIUS, n_row: int = 80, n_turn: int = 160):
    """The path of Fig. 2 as an (N, 2) array of points, rows alternating direction."""
    if turn_radius <= spacing / 2:
        raise ValueError("a bulb turn needs a radius larger than half the row spacing")
    dx = float(np.sqrt(turn_radius ** 2 - (spacing / 2) ** 2))
    pts = []
    for i in range(n_rows):
        y = -i * spacing
        xs = np.linspace(0.0, row_length, n_row) if i % 2 == 0 else \
            np.linspace(row_length, 0.0, n_row)
        pts.append(np.stack([xs, np.full_like(xs, y)], axis=1))
        if i + 1 == n_rows:
            break
        right = i % 2 == 0
        y_mid = y - spacing / 2
        cx = row_length + dx if right else -dx
        a0 = np.arctan2(y - y_mid, (row_length if right else 0.0) - cx)
        a1 = np.arctan2(y - spacing - y_mid, (row_length if right else 0.0) - cx)
        # The long way round the circle is what makes this a bulb and not a corner: the
        # vehicle leaves the row, goes over the top, round the outside and back underneath.
        # At the right-hand end that is clockwise (negative sweep), at the left-hand end
        # anticlockwise. Taking the short arc instead collapses the turn onto the rows.
        sweep = -((a0 - a1) % (2 * np.pi)) if right else ((a1 - a0) % (2 * np.pi))
        ang = a0 + np.linspace(0.0, sweep, n_turn)
        pts.append(np.stack([cx + turn_radius * np.cos(ang),
                             y_mid + turn_radius * np.sin(ang)], axis=1))
    return np.concatenate(pts, axis=0)


def agricultural(**kw) -> "nmppfc.path.Path":
    """Fig. 2 as a Path parameterised by arc length; ``gamma_range`` is (0, total length)."""
    pts = polyline(**kw)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])

    def xy(g):
        g = float(np.clip(g, 0.0, s[-1]))
        return float(np.interp(g, s, pts[:, 0])), float(np.interp(g, s, pts[:, 1]))

    return nmppfc.path.Path(xy=xy, gamma_range=(0.0, float(s[-1])),
                            name="agricultural (Fig. 2)", closed=False)


def length(**kw) -> float:
    pts = polyline(**kw)
    return float(np.hypot(*np.diff(pts, axis=0).T).sum())
