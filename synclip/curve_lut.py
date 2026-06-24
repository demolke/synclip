"""Monotone-cubic response-curve baking.

The interactive editor lives in ``ui/curve_editor.py``, but the curve it bakes
(a 0..1 -> 0..1 lookup table) is also applied by the headless modifier pipeline.
Keeping the baking here lets the engine (modifiers, view_pipeline) stay Qt-free
while the UI widget imports the same function for its drawing/emit path.
"""

from __future__ import annotations

import math

LUT_SIZE = 64


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _monotone_tangents(xs: list[float], ys: list[float]) -> list[float]:
    """Fritsch-Carlson monotone-cubic tangents for the given knots.

    Shared by the response-curve LUT and the animation sampler so both draw the
    same smooth, overshoot-free curve through the control points.
    """
    k = len(xs)
    if k == 1:
        return [0.0]
    d = [(ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(k - 1)]
    m = [0.0] * k
    m[0] = d[0]
    m[k - 1] = d[k - 2]
    for i in range(1, k - 1):
        m[i] = 0.0 if d[i - 1] * d[i] <= 0 else (d[i - 1] + d[i]) / 2.0
    for i in range(k - 1):
        if d[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / d[i]
            b = m[i + 1] / d[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / math.sqrt(s)
                m[i] = t * a * d[i]
                m[i + 1] = t * b * d[i]
    return m


def _hermite_eval(xs: list[float], ys: list[float], m: list[float], x: float) -> float:
    """Evaluate the cubic Hermite spline (knots xs/ys, tangents m) at xs[0]<x<xs[-1]."""
    k = len(xs)
    i = 0
    while i < k - 1 and xs[i + 1] < x:
        i += 1
    h = xs[i + 1] - xs[i]
    if h == 0:
        return ys[i + 1]
    t = (x - xs[i]) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    return h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]


def sample_curve(points: list, t: float) -> float:
    """Smooth (monotone-cubic) sample of *points* at x=*t*, returning the raw y.

    Uses the same Fritsch-Carlson spline as :func:`build_lut` so the rendered
    animation curve and the values applied by the engine match, but it does NOT
    clamp the output (an influence offset curve may go negative) and does not
    require the y values to be monotonic. x outside the point range holds the
    nearest end. With only two points the spline reduces to a straight line.
    """
    if not points:
        return 0.0
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if t <= xs[0]:
        return float(ys[0])
    if t >= xs[-1]:
        return float(ys[-1])
    m = _monotone_tangents(xs, ys)
    return float(_hermite_eval(xs, ys, m, t))


def build_lut(points: list[tuple[float, float]], n: int = LUT_SIZE) -> list[float]:
    """Bake *points* into an n-entry monotone-cubic LUT over x in [0, 1]."""
    pts = sorted(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    k = len(pts)
    if k == 1:
        return [clamp01(ys[0])] * n

    m = _monotone_tangents(xs, ys)
    lut: list[float] = []
    for j in range(n):
        x = j / (n - 1)
        if x <= xs[0]:
            lut.append(clamp01(ys[0]))
        elif x >= xs[-1]:
            lut.append(clamp01(ys[-1]))
        else:
            lut.append(clamp01(_hermite_eval(xs, ys, m, x)))
    return lut
