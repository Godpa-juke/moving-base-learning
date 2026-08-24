from __future__ import annotations

import math
import unittest

from marine_manipulator.trajectory import ping_pong_line_offset


class PrecisionStartTrajectoryTests(unittest.TestCase):
    def test_holds_at_negative_endpoint_for_one_second(self) -> None:
        for t in (0.0, 0.25, 0.999):
            self.assertAlmostEqual(ping_pong_line_offset(t, amplitude=0.10, hold_s=1.0, traverse_s=4.0), -0.10)

    def test_moves_smoothly_to_opposite_endpoint_then_returns(self) -> None:
        self.assertAlmostEqual(ping_pong_line_offset(1.0, 0.10, 1.0, 4.0), -0.10)
        self.assertAlmostEqual(ping_pong_line_offset(3.0, 0.10, 1.0, 4.0), 0.0, places=7)
        self.assertAlmostEqual(ping_pong_line_offset(5.0, 0.10, 1.0, 4.0), 0.10, places=7)
        self.assertAlmostEqual(ping_pong_line_offset(7.0, 0.10, 1.0, 4.0), 0.0, places=7)
        self.assertAlmostEqual(ping_pong_line_offset(9.0, 0.10, 1.0, 4.0), -0.10, places=7)

    def test_has_zero_velocity_at_both_endpoints(self) -> None:
        eps = 1.0e-4
        near_start = ping_pong_line_offset(1.0 + eps, 0.10, 1.0, 4.0)
        near_end = ping_pong_line_offset(5.0 - eps, 0.10, 1.0, 4.0)
        self.assertLess(abs(near_start + 0.10), 1.0e-7)
        self.assertLess(abs(near_end - 0.10), 1.0e-7)


if __name__ == "__main__":
    unittest.main()
