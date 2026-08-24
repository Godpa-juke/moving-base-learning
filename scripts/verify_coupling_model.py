#!/usr/bin/env python3
"""Check the dynamic model the coupled baseline is about to be built on.

Baseline B predicts what its own joint command does to the hull it is standing on. The
prediction rests on two tensors PhysX exposes and on two conventions the documentation
leaves ambiguous: whether the generalized mass matrix orders the root degrees of freedom
first, and whether its root block and the Jacobian's root columns are expressed about
the link frame or the centre of mass, in world axes.

Guessing wrong would not raise an error. It would produce a controller that is subtly
mis-compensating and would then be reported as 'model-based control does not help here',
which is exactly the class of unverified mechanism ``docs/UVMS_PLAN.md`` forbids. So the
conventions are measured instead:

* the top-left 3x3 of the root block must be the total system mass on the diagonal;
* the reaction predicted from momentum conservation,
  ``dv_b = -H_b^-1 H_bm dq_dot``, must match the hull velocity change PhysX actually
  produces when the arm is commanded to move and nothing else acts on the hull.

The second check is the real one. It is a closed-loop test of the whole partitioning: an
ordering error, a frame error or a transpose error all break it.
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
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=971)
parser.add_argument("--run-name", default="uvms_coupling_model")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    import isaaclab_tasks  # noqa: F401
    import marine_manipulator.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from marine_manipulator import provenance
    from marine_manipulator.tasks.random_base_line import mdp

    device = args.device or "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs, use_fabric=True)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    env.reset()
    # The disturbance would add an unmodelled impulse to every step and blur the very
    # quantity being measured, so the hull is left with only the arm acting on it. The
    # reset launches the hull onto the wave's trajectory, so its velocity is cleared too.
    mdp._state(env)["amplitude"][:] = 0.0
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))
    robot.update(0.0)

    view = robot.root_physx_view
    mass_matrix = view.get_generalized_mass_matrices()
    jacobians = view.get_jacobians()
    total_mass = float(view.get_masses()[0].sum())
    coms = view.get_coms()

    root_block = mass_matrix[0, :6, :6]
    diagonal_mass = float(root_block[:3, :3].diagonal().mean())
    off_diagonal = float(root_block[:3, :3].abs().sum() - root_block[:3, :3].diagonal().abs().sum())

    wrist_index = robot.body_names.index("wrist_3_link")
    base_index = robot.body_names.index("base_link")

    from isaaclab.utils.math import quat_apply

    # Whether the root block is taken about the system centre of mass or about the base
    # link's own frame decides which velocity the prediction has to be compared against.
    # For a spatial inertia about a point offset from the centre of mass, the
    # translation-rotation coupling block is -m*skew(r); about the centre of mass it is
    # zero. base_link's own centre of mass offset is separately reported above.
    coupling_block = float(mass_matrix[0, :3, 3:6].abs().max())

    # The Jacobian's own conventions, read off the base link: its rows against the root
    # columns must be the identity (the root link moves exactly with the root degrees of
    # freedom) and its columns against the joints must vanish (bending the arm does not
    # move the base link within the articulation's own frame). This pins the linear-
    # before-angular row order and the root-before-joint column order in one check.
    base_jacobian = jacobians[0, base_index]
    jacobian_root_identity_error = float(
        (base_jacobian[:, :6] - torch.eye(6, device=base_jacobian.device)).abs().max()
    )
    jacobian_base_joint_coupling = float(base_jacobian[:, 6:].abs().max())

    predictions = []
    reaction_only = []
    displacement_predictions = {0.5: [], 1.0: []}
    displacement_actual = []
    actuals = []
    action = torch.zeros((env.num_envs, 6), device=device)
    for step in range(args.steps):
        # A command large enough that the rate limiter saturates, reversing partway, so
        # the joint velocity actually changes rather than sitting at a constant sweep
        # where the reaction has already been absorbed.
        action[:] = 3.0 if step < args.steps // 2 else -3.0

        matrices = view.get_generalized_mass_matrices()
        h_b = matrices[:, :6, :6]
        h_bm = matrices[:, :6, 6:]
        before_vel = torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=1).clone()
        before_joint_vel = robot.data.joint_vel.clone()
        # The wrench the environment is about to apply, rotated into the world axes the
        # mass matrix is expressed in. Leaving it out is what a naive momentum argument
        # does, and the gap between the two predictions is how much it costs.
        force_b, torque_b = mdp.hydrodynamic_wrench(env)
        quat = robot.data.root_quat_w
        wrench_w = torch.cat((quat_apply(quat, force_b), quat_apply(quat, torque_b)), dim=1)
        before_pos = robot.data.root_pos_w.clone()

        env.step(action)

        after_vel = torch.cat((robot.data.root_lin_vel_w, robot.data.root_ang_vel_w), dim=1)
        delta_joint_vel = robot.data.joint_vel - before_joint_vel
        reaction = -(h_bm @ delta_joint_vel.unsqueeze(2)).squeeze(2)
        impulse = wrench_w * env.step_dt
        predictions.append(torch.linalg.solve(h_b, (reaction + impulse).unsqueeze(2)).squeeze(2))
        reaction_only.append(torch.linalg.solve(h_b, reaction.unsqueeze(2)).squeeze(2))
        actuals.append((after_vel - before_vel).clone())

        # How much of the velocity step lands inside the same control step is the one
        # coefficient the coupled controller cannot derive: PhysX integrates semi-
        # implicitly (all of it, factor 1.0) but the joint velocity itself ramps under
        # the PD (half of it, factor 0.5). Both are predicted here and the better one is
        # fixed before any controller runs, rather than tuned after seeing a result.
        delta_vel = torch.linalg.solve(h_b, (reaction + impulse).unsqueeze(2)).squeeze(2)
        for factor in displacement_predictions:
            step_vel = before_vel + factor * delta_vel
            displacement_predictions[factor].append(step_vel[:, :3] * env.step_dt)
        displacement_actual.append(robot.data.root_pos_w - before_pos)

    displacement_actual = torch.cat(displacement_actual)
    predicted = torch.cat(predictions)
    predicted_reaction_only = torch.cat(reaction_only)
    actual = torch.cat(actuals)
    # Compare only where there is something to compare: steps where the arm barely
    # accelerated carry no signal and would flatter the correlation.
    active = actual.norm(dim=1) > 1e-4
    predicted = predicted[active]
    predicted_reaction_only = predicted_reaction_only[active]
    actual = actual[active]

    def agreement(a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity of the two flattened signals, in [-1, 1]."""
        a, b = a.flatten().double(), b.flatten().double()
        return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))

    scale = float(predicted.norm() / actual.norm().clamp(min=1e-12))
    checks = {
        "root_dofs_come_first": abs(diagonal_mass - total_mass) < 0.05 * total_mass,
        "root_block_is_diagonal_in_translation": off_diagonal < 0.01 * total_mass,
        "reaction_direction_matches": agreement(predicted, actual) > 0.9,
        "external_wrench_is_needed": (
            agreement(predicted, actual) > agreement(predicted_reaction_only, actual)
        ),
        "reaction_magnitude_matches": 0.7 < scale < 1.4,
        "jacobian_root_columns_are_identity": jacobian_root_identity_error < 1e-4,
        "jacobian_base_row_ignores_joints": jacobian_base_joint_coupling < 1e-4,
    }
    displacement_error = {
        str(factor): float(
            (torch.cat(values) - displacement_actual).norm() / displacement_actual.norm()
        )
        for factor, values in displacement_predictions.items()
    }

    project = Path(os.environ["MARINE_PROJECT_DIR"])
    out_dir = project / "outputs" / "evaluation" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "probe": "uvms_coupling_model",
        "task": args.task,
        "provenance": provenance.record(project, out_dir),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "mass_matrix_shape": list(mass_matrix.shape),
        "jacobian_shape": list(jacobians.shape),
        "body_names": robot.body_names,
        "base_body_index": base_index,
        "wrist_body_index": wrist_index,
        "base_com_b": coms[0, base_index].cpu().tolist(),
        "total_mass_kg": total_mass,
        "root_block_diagonal_mass_kg": diagonal_mass,
        "root_block_translation_off_diagonal": off_diagonal,
        "compared_samples": int(predicted.shape[0]),
        "root_translation_rotation_coupling": coupling_block,
        "jacobian_root_identity_error": jacobian_root_identity_error,
        "jacobian_base_joint_coupling": jacobian_base_joint_coupling,
        "displacement_relative_error_by_reaction_factor": displacement_error,
        "reaction_cosine_similarity": agreement(predicted, actual),
        "reaction_only_cosine_similarity": agreement(predicted_reaction_only, actual),
        "reaction_magnitude_ratio": scale,
        "checks": checks,
        "status": "coupling_model_ok" if all(checks.values()) else "coupling_model_failed",
    }
    env.close()
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("uvms_coupling_model_summary", json.dumps(result, sort_keys=True), flush=True)
    for name, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}", flush=True)
except BaseException:
    import traceback

    traceback.print_exc()
    os._exit(1)
else:
    app.close()
