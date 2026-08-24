#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    from test_precision_far_immediate_contract import (
        test_far_precision_task_ids_are_distinct_and_registered,
        test_far_start_uses_new_ik_and_immediate_neutral_start_motion,
        test_line_is_shifted_half_meter_opposite_world_red_x_arrow,
        test_target_still_holds_one_second,
    )

    test_far_precision_task_ids_are_distinct_and_registered()
    test_line_is_shifted_half_meter_opposite_world_red_x_arrow()
    test_far_start_uses_new_ik_and_immediate_neutral_start_motion()
    test_target_still_holds_one_second()
    print("precision_far_immediate_contract PASS", flush=True)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()
