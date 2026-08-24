from __future__ import annotations

import math
import random
from dataclasses import dataclass

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class AxisRange:
    amplitude: tuple[float, float]
    frequency: tuple[float, float]


@dataclass(frozen=True)
class AxisParameters:
    amplitude: float
    frequency: float
    phase: float


MOTION_RANGES = {
    "x": AxisRange((0.02, 0.06), (0.10, 0.30)),
    "y": AxisRange((0.02, 0.05), (0.10, 0.30)),
    "z": AxisRange((0.03, 0.08), (0.10, 0.25)),
    "roll": AxisRange((0.02, 0.06), (0.08, 0.22)),
    "pitch": AxisRange((0.02, 0.06), (0.08, 0.22)),
    "yaw": AxisRange((0.01, 0.04), (0.06, 0.18)),
}


def sample_episode_parameters(num_envs: int, seed: int) -> list[dict[str, AxisParameters]]:
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    rng = random.Random(seed)
    return [
        {
            axis: AxisParameters(
                amplitude=rng.uniform(*MOTION_RANGES[axis].amplitude),
                frequency=rng.uniform(*MOTION_RANGES[axis].frequency),
                phase=rng.uniform(0.0, 2.0 * math.pi),
            )
            for axis in AXES
        }
        for _ in range(num_envs)
    ]
