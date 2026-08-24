from __future__ import annotations

import math


def ping_pong_line_offset(time_s: float, amplitude: float, hold_s: float, traverse_s: float) -> float:
    """Hold at -amplitude, then cosine-ease between both endpoints forever."""
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")
    if hold_s < 0.0:
        raise ValueError("hold_s must be non-negative")
    if traverse_s <= 0.0:
        raise ValueError("traverse_s must be positive")
    if time_s <= hold_s:
        return -amplitude
    phase = math.pi * (time_s - hold_s) / traverse_s
    return -amplitude * math.cos(phase)
