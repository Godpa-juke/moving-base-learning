#!/usr/bin/env python3
"""Stage 1e: check the hydrodynamic model does what it claims, before any controller.

Four probes, each designed so that a plausible implementation bug makes it fail rather
than merely look different. Running the task and finding the tracking error 'reasonable'
would not distinguish a working drag term from one that is silently never applied.

* ``settle``   - zero command, no waves, arm still. The hull must not drift and must not
                 spin. This is the equilibrium the other probes are perturbations of.
* ``righting`` - the hull is released at 0.3 rad of roll with waves off. The centre of
                 buoyancy sitting above the centre of gravity must bring it back to
                 level. If the restoring moment is missing or has the wrong sign, the
                 hull stays tilted or falls over.
* ``nodrag``   - the same wave forcing with the drag coefficients set to zero. The hull
                 must move *further* than with drag. If the two are equal the drag
                 wrench is not reaching the simulation at all, which is the failure the
                 plan singles out.
* ``reaction`` - a fast arm motion with waves off. Whatever the hull does here is the
                 arm's own reaction, and it is the honest version of the Stage 0 number:
                 a vehicle with a real inertia, real added mass, and drag resisting it.

The probes run on the registered UVMS task rather than a throwaway scene, so what is
verified is the thing the experiments will use.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Marine-UR3-Uvms-WorldLineFreeFloating-Play-v0")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--seed", type=int, default=970)
parser.add_argument("--run-name", default="uvms_verify")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from marine_manipulator import provenance
    from marine_manipulator.tasks.random_base_line import mdp

    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    params = mdp.vehicle_params(env)
    steps = int(args.seconds / env.step_dt)

    def zero_waves() -> None:
        """Silence the disturbance without removing the sampler.

        Amplitude is what `hydrodynamics.wave_wrench` scales the force by, so zeroing it
        leaves every other code path - the sampling event, the state dictionary, the
        per-step evaluation - exactly as the real task runs them.

        The root velocity has to go with it. `reset_uvms_root` launches the hull onto
        the wave's trajectory, so a hull whose waves are silenced after the reset would
        still be coasting at the velocity the silenced wave asked for, and the settling
        probe would measure that coast rather than the equilibrium it is checking.
        """
        state = mdp._state(env)
        state["amplitude"][:] = 0.0
        robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
        robot.update(0.0)

    def rollout(
        waves: bool,
        drag: bool,
        initial_roll_rad: float = 0.0,
        arm_speed: bool = False,
    ) -> dict[str, float]:
        env.reset()
        if not waves:
            zero_waves()
        if not drag:
            # Zeroing the coefficients rather than skipping the call keeps the wrench
            # path itself under test: if the hull still slows down, something else is
            # damping it and the drag measurement would be misattributed.
            params_zero = params.scaled_drag(0.0)
            mdp.vehicle_params(env)  # ensure cached
            setattr(env, mdp._VEHICLE_ATTR, params_zero)
        else:
            setattr(env, mdp._VEHICLE_ATTR, params)

        origin = robot.data.default_root_state[:, :3] + env.scene.env_origins
        if initial_roll_rad:
            root_pose = robot.data.root_state_w[:, :7].clone()
            roll = torch.full((env.num_envs,), initial_roll_rad, device=env.device)
            zero = torch.zeros_like(roll)
            root_pose[:, 3:7] = quat_mul(root_pose[:, 3:7], quat_from_euler_xyz(roll, zero, zero))
            robot.write_root_pose_to_sim(root_pose)
            robot.update(0.0)

        action = torch.zeros((env.num_envs, 6), device=env.device)
        peak_translation = torch.zeros(env.num_envs, device=env.device)
        peak_roll = torch.zeros(env.num_envs, device=env.device)
        final_roll = torch.zeros(env.num_envs, device=env.device)
        for step in range(steps):
            if arm_speed:
                # Full-scale command, reversing halfway: the rate limiter turns this
                # into a sweep at the joint speed limit, which is where the reaction
                # torque peaks.
                action[:] = 3.0 if step < steps // 2 else -3.0
            env.step(action)
            offset = mdp.uvms_base_pose(env)
            peak_translation = torch.maximum(
                peak_translation, torch.linalg.norm(robot.data.root_pos_w - origin, dim=1)
            )
            peak_roll = torch.maximum(peak_roll, offset[:, 3].abs())
            final_roll = offset[:, 3]

        setattr(env, mdp._VEHICLE_ATTR, params)
        return {
            "peak_translation_m": float(peak_translation.mean()),
            "peak_roll_rad": float(peak_roll.mean()),
            "final_roll_rad": float(final_roll.abs().mean()),
        }

    probes = {
        "settle": rollout(waves=False, drag=True),
        "righting": rollout(waves=False, drag=True, initial_roll_rad=0.3),
        "waves": rollout(waves=True, drag=True),
        "waves_nodrag": rollout(waves=True, drag=False),
        "reaction": rollout(waves=False, drag=True, arm_speed=True),
    }

    checks = {
        # A neutral hull with no disturbance and a still arm has nothing acting on it.
        "settles_without_drift": probes["settle"]["peak_translation_m"] < 1e-3,
        # 0.3 rad released must decay, not persist and not diverge.
        "rights_itself": probes["righting"]["final_roll_rad"] < 0.15,
        "righting_is_bounded": probes["righting"]["peak_roll_rad"] < 0.6,
        # The decisive one: without drag the same forcing must move the hull further.
        "drag_resists_waves": (
            probes["waves_nodrag"]["peak_translation_m"] > 1.05 * probes["waves"]["peak_translation_m"]
        ),
        # And the arm must be able to move the hull, or the round has no question in it.
        "arm_moves_hull": probes["reaction"]["peak_translation_m"] > 1e-3,
    }

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "probe": "uvms_verify",
        "task": args.task,
        "provenance": provenance.record(project, out_dir),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "seconds": args.seconds,
        "vehicle": {
            "mass_kg": params.mass_kg,
            "inertia_kg_m2": list(params.inertia_kg_m2),
            "effective_mass_kg": params.effective_mass_kg,
            "effective_inertia_kg_m2": list(params.effective_inertia_kg_m2),
            "cob_offset_b": list(params.cob_offset_b),
        },
        "probes": probes,
        "checks": checks,
        "status": "uvms_verify_ok" if all(checks.values()) else "uvms_verify_failed",
    }
    env.close()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("uvms_verify_summary", json.dumps(result, sort_keys=True), flush=True)
    for name, values in probes.items():
        print(
            f"  {name:14s} translation {values['peak_translation_m'] * 1000:8.2f} mm"
            f"   peak_roll {values['peak_roll_rad'] * 1000:8.2f} mrad"
            f"   final_roll {values['final_roll_rad'] * 1000:8.2f} mrad",
            flush=True,
        )
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()
