#!/usr/bin/env python3
"""Stage 0 go/no-go: does the arm's own motion move a free-floating vehicle?

The previous round drove the base kinematically, which made it an exogenous signal
and the task a pure inversion. This round only has a question in it if the arm can
actually push the hull around. That is a property of the mass ratio and nothing
else, so it is measured before anything is built on top of it.

The scene is deliberately throwaway: a free-root UR3 whose ``base_link`` carries the
vehicle's mass, no hydrodynamics, no waves, no controller. One environment per
candidate vehicle mass, all stepped together.

Two numbers per mass:

* ``hull_disp_m`` - how far the hull is displaced by the arm swinging.
* ``tcp_disp_from_hull_m`` - how far the tool tip is dragged *by the hull moving*,
  which is the part a controller would have to reject. It is the difference between
  where the tip actually is and where it would be if the hull had never moved, so
  the arm's own commanded motion cancels out of it.

With ``--gravity`` the arm also falls, and the fall dwarfs the reaction; the
still-arm control run is then subtracted to isolate the coupling. The default
(zero gravity) is the honest reading, because the vehicle this models is designed
to be near neutrally buoyant.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--masses",
    default="50,100,200,500,2000",
    help="base_link masses to sweep, kg. One simulated environment per entry.",
)
parser.add_argument(
    "--gravity",
    action="store_true",
    help="run with gravity enabled. The arm then falls as well as reacting, so a "
    "still-arm control run is subtracted; the default zero-gravity case is the "
    "neutrally buoyant vehicle this study is actually about.",
)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--run-name", default="stage0_coupling")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply, quat_error_magnitude

    from marine_manipulator import calibration, provenance, ur3_kin, uvms_asset

    masses = [float(value) for value in args.masses.split(",")]
    device = args.device or "cuda:0"
    physics_dt = 1.0 / 120.0

    # The base is released by moving the articulation root onto `base_link`; see
    # `uvms_asset`. Per-environment masses are written through the PhysX view below
    # rather than through the spawner, because one scene has to hold the whole sweep.
    robot_cfg = uvms_asset.free_floating_ur3_cfg()
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    robot_cfg.spawn.rigid_props.disable_gravity = not args.gravity
    robot_cfg.init_state = robot_cfg.init_state.replace(
        pos=(0.0, 0.0, 1.0),
        joint_pos=dict(zip(ur3_kin.JOINT_NAMES, calibration.BOX_CENTER_JOINT_POS)),
    )

    @configclass
    class Stage0SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=physics_dt, device=device, gravity=(0.0, 0.0, -9.81 if args.gravity else 0.0))
    )
    scene = InteractiveScene(Stage0SceneCfg(num_envs=len(masses), env_spacing=6.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    base_index = robot.body_names.index("base_link")
    wrist_index = robot.body_names.index("wrist_3_link")

    body_masses = robot.root_physx_view.get_masses().clone()
    arm_mass_kg = float(body_masses[0].sum())
    base_mass_kg = float(body_masses[0, base_index])

    # Scale the base link's inertia with its mass so the swap reads as a denser body
    # of the same shape, not a heavy body with a pencil's rotational inertia. Both the
    # mass and the inertia are the vehicle's stand-in; Stage 1 replaces them with real
    # ROV values, and lumping the hull into base_link is a stated shortcut either way.
    inertias = robot.root_physx_view.get_inertias().clone()
    for env_index, mass in enumerate(masses):
        ratio = mass / base_mass_kg
        body_masses[env_index, base_index] = mass
        inertias[env_index, base_index] *= ratio
    env_indices = torch.arange(len(masses), dtype=torch.int32)
    robot.root_physx_view.set_masses(body_masses, env_indices)
    robot.root_physx_view.set_inertias(inertias, env_indices)

    speed = torch.tensor(calibration.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=device)
    start_q = torch.tensor(calibration.BOX_CENTER_JOINT_POS, device=device).repeat(len(masses), 1)
    tool = calibration.TCP_OFFSET[2]

    def run(moving: bool) -> dict[str, list[float]]:
        """Step the scene; return the peak hull and hull-induced TCP displacements."""
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim(root_state[:, :7])
        robot.write_root_velocity_to_sim(root_state[:, 7:])
        robot.write_joint_state_to_sim(start_q, torch.zeros_like(start_q))
        robot.reset()
        scene.write_data_to_sim()
        sim.step()
        scene.update(physics_dt)

        origin_pos = robot.data.body_pos_w[:, base_index].clone()
        origin_quat = robot.data.body_quat_w[:, base_index].clone()

        peak_hull = torch.zeros(len(masses), device=device)
        peak_hull_rot = torch.zeros(len(masses), device=device)
        peak_tcp = torch.zeros(len(masses), device=device)

        steps = int(args.seconds / physics_dt)
        # Sweep every joint at its rate limit, reversing halfway. The reversal is
        # where the reaction torque peaks, which is the number this probe is after.
        for step in range(steps):
            t = step * physics_dt
            direction = 1.0 if t < args.seconds / 2.0 else -1.0
            travelled = t if direction > 0 else args.seconds - t
            target = start_q + speed * travelled if moving else start_q.clone()
            robot.set_joint_position_target(target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(physics_dt)

            hull_pos = robot.data.body_pos_w[:, base_index]
            hull_quat = robot.data.body_quat_w[:, base_index]
            peak_hull = torch.maximum(peak_hull, torch.linalg.norm(hull_pos - origin_pos, dim=1))
            peak_hull_rot = torch.maximum(peak_hull_rot, quat_error_magnitude(hull_quat, origin_quat))

            # Where the tip would be if the hull had never moved: the same joint
            # configuration, placed on the hull's initial pose. The difference is
            # purely the disturbance the hull hands to the tool.
            tcp_root = ur3_kin.tcp_position(robot.data.joint_pos, tool)
            tcp_if_still = origin_pos + quat_apply(origin_quat, tcp_root)
            tcp_now = robot.data.body_pos_w[:, wrist_index] + quat_apply(
                robot.data.body_quat_w[:, wrist_index],
                torch.tensor(calibration.TCP_OFFSET, device=device).repeat(len(masses), 1),
            )
            peak_tcp = torch.maximum(peak_tcp, torch.linalg.norm(tcp_now - tcp_if_still, dim=1))

        return {
            "hull_disp_m": peak_hull.cpu().tolist(),
            "hull_rot_rad": peak_hull_rot.cpu().tolist(),
            "tcp_disp_from_hull_m": peak_tcp.cpu().tolist(),
        }

    moving = run(moving=True)
    still = run(moving=False)

    rows = []
    for index, mass in enumerate(masses):
        rows.append(
            {
                "vehicle_mass_kg": mass,
                "mass_ratio_arm_over_vehicle": arm_mass_kg / mass,
                "hull_disp_m": moving["hull_disp_m"][index],
                "hull_rot_rad": moving["hull_rot_rad"][index],
                "tcp_disp_from_hull_m": moving["tcp_disp_from_hull_m"][index],
                "still_hull_disp_m": still["hull_disp_m"][index],
                "still_tcp_disp_from_hull_m": still["tcp_disp_from_hull_m"][index],
                # With gravity on the fall is common to both runs and subtracts out.
                "coupling_hull_disp_m": moving["hull_disp_m"][index] - still["hull_disp_m"][index],
                "coupling_tcp_disp_m": (
                    moving["tcp_disp_from_hull_m"][index] - still["tcp_disp_from_hull_m"][index]
                ),
            }
        )

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "probe": "stage0_coupling",
        "provenance": provenance.record(project, out_dir),
        "gravity": args.gravity,
        "seconds": args.seconds,
        "physics_dt": physics_dt,
        "arm_total_mass_kg": arm_mass_kg,
        "stock_base_link_mass_kg": base_mass_kg,
        "body_names": robot.body_names,
        "stock_body_masses_kg": body_masses[0].cpu().tolist(),
        "rows": rows,
        "status": "stage0_coupling_ok",
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("stage0_coupling_summary", json.dumps(result, sort_keys=True), flush=True)
    for row in rows:
        print(
            f"  mass {row['vehicle_mass_kg']:7.1f} kg   hull {row['coupling_hull_disp_m'] * 1000:8.2f} mm"
            f"   hull_rot {row['hull_rot_rad'] * 1000:8.2f} mrad"
            f"   tcp {row['coupling_tcp_disp_m'] * 1000:8.2f} mm",
            flush=True,
        )
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()
