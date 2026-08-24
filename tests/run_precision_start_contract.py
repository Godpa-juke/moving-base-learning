#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    from test_precision_start_contract import (
        test_line_starts_at_one_endpoint_holds_then_traverses,
        test_precision_reward_keeps_broad_pursuit_and_adds_one_cm_terms,
        test_precision_start_task_ids_are_registered,
        test_reset_uses_verified_ik_pose_and_neutral_base_start,
    )

    test_precision_start_task_ids_are_registered()
    test_line_starts_at_one_endpoint_holds_then_traverses()
    test_reset_uses_verified_ik_pose_and_neutral_base_start()
    test_precision_reward_keeps_broad_pursuit_and_adds_one_cm_terms()
    print("precision_start_contract PASS", flush=True)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()
