from __future__ import annotations

import math
import unittest

from marine_manipulator.motion_config import AXES, MOTION_RANGES, sample_episode_parameters


class RandomBaseMotionContractTests(unittest.TestCase):
    def test_all_six_axes_are_randomized_with_positive_ranges(self) -> None:
        self.assertEqual(AXES, ("x", "y", "z", "roll", "pitch", "yaw"))
        self.assertEqual(set(MOTION_RANGES), set(AXES))
        for axis in AXES:
            cfg = MOTION_RANGES[axis]
            self.assertLess(cfg.amplitude[0], cfg.amplitude[1])
            self.assertGreater(cfg.amplitude[0], 0.0)
            self.assertLess(cfg.frequency[0], cfg.frequency[1])
            self.assertGreater(cfg.frequency[0], 0.0)

    def test_episode_sampler_is_seeded_bounded_and_varies_by_environment(self) -> None:
        first = sample_episode_parameters(num_envs=4, seed=818)
        again = sample_episode_parameters(num_envs=4, seed=818)
        other = sample_episode_parameters(num_envs=4, seed=819)

        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 4)
        self.assertGreater(len({tuple(env[axis].phase for axis in AXES) for env in first}), 1)
        for env in first:
            self.assertEqual(set(env), set(AXES))
            for axis in AXES:
                value = env[axis]
                cfg = MOTION_RANGES[axis]
                self.assertGreaterEqual(value.amplitude, cfg.amplitude[0])
                self.assertLessEqual(value.amplitude, cfg.amplitude[1])
                self.assertGreaterEqual(value.frequency, cfg.frequency[0])
                self.assertLessEqual(value.frequency, cfg.frequency[1])
                self.assertGreaterEqual(value.phase, 0.0)
                self.assertLessEqual(value.phase, 2.0 * math.pi)


if __name__ == "__main__":
    unittest.main()
