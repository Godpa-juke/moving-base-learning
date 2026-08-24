#!/usr/bin/env python3
from __future__ import annotations

import os
import traceback
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
try:
    from test_target_conditioned_contract import (
        test_actor_observes_explicit_world_target_error,
        test_base_moves_immediately_and_reward_tracks_same_command,
        test_target_conditioned_task_is_registered,
        test_each_episode_samples_a_reachable_random_line_in_the_negative_x_band,
        test_rendered_line_endpoints_follow_sampled_direction,
        test_realistic_joint_target_rate_limit,
        test_reward_engineering_includes_realistic_speed_and_multiscale_tracking,
    )
    test_target_conditioned_task_is_registered()
    test_each_episode_samples_a_reachable_random_line_in_the_negative_x_band()
    test_actor_observes_explicit_world_target_error()
    test_rendered_line_endpoints_follow_sampled_direction()
    test_realistic_joint_target_rate_limit()
    test_reward_engineering_includes_realistic_speed_and_multiscale_tracking()
    test_base_moves_immediately_and_reward_tracks_same_command()
    print("target_conditioned_contract PASS", flush=True)
except BaseException:
    traceback.print_exc()
    os._exit(1)
else:
    app.close()
