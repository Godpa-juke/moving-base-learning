#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

try:
    from test_random_base_line_contract import (
        test_all_six_base_axes_are_randomized_at_every_reset,
        test_fresh_task_ids_are_registered,
        test_initial_reward_is_broad_pursuit_without_smoothness_confounds,
        test_policy_observes_normalized_random_base_state_and_world_tcp,
        test_task_uses_stock_ur3_and_exact_cylinder_tip_tcp,
    )

    test_fresh_task_ids_are_registered()
    test_task_uses_stock_ur3_and_exact_cylinder_tip_tcp()
    test_all_six_base_axes_are_randomized_at_every_reset()
    test_initial_reward_is_broad_pursuit_without_smoothness_confounds()
    test_policy_observes_normalized_random_base_state_and_world_tcp()
    print("fresh_random_base_line_contract PASS", flush=True)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()
