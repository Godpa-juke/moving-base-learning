"""The rollout and the metrics, extracted so more than one script can share them.

``scripts/evaluate.py`` measures one condition per process. The free-floating round's
mismatch sweep measures dozens, and paying Isaac Sim's start-up for each one would cost
more than the rollouts do. Rather than let the sweep grow its own copy of the metric
code — where a threshold or a phase rule could quietly drift away from the numbers the
rest of the study reports — both call in here.

The one substantive rule encoded below is the phase split. An episode begins with the
tool already on the line, but a controller still needs a step or two to settle, and any
sample taken before the tool is within ``capture_threshold`` is transit rather than
tracking. Averaging the two together is what reported a 6 mm policy as 0.098 m in the
previous round (``docs/FINDINGS.md``), so ``track_`` is the number that means what the
task is about and ``all_`` is kept only for continuity.
"""

from __future__ import annotations

import torch


def stats(values: torch.Tensor, prefix: str) -> dict:
    """Error distribution summary in metres, with success rates at fixed thresholds."""
    if values.numel() == 0:
        return {f"{prefix}samples": 0}
    values = values.double()
    return {
        f"{prefix}samples": int(values.numel()),
        f"{prefix}rmse_m": float(values.square().mean().sqrt()),
        f"{prefix}mean_error_m": float(values.mean()),
        f"{prefix}p95_error_m": float(values.quantile(0.95)),
        f"{prefix}max_error_m": float(values.max()),
        f"{prefix}success_1mm": float((values <= 0.001).double().mean()),
        f"{prefix}success_2mm": float((values <= 0.002).double().mean()),
        f"{prefix}success_5mm": float((values <= 0.005).double().mean()),
        f"{prefix}success_1cm": float((values <= 0.01).double().mean()),
        f"{prefix}success_5cm": float((values <= 0.05).double().mean()),
        f"{prefix}success_10cm": float((values <= 0.10).double().mean()),
    }


def rollout(env, policy, steps: int, capture_threshold: float, tcp_offset) -> dict:
    """Step ``policy`` through ``env`` and return the tracking metrics.

    ``env`` is the RSL-RL wrapper; ``env.unwrapped`` is the manager-based environment.
    The tool pose is recomputed here from the wrist link rather than read from a reward
    term, so the metric does not depend on which reward terms a task happens to define.
    """
    from isaaclab.utils.math import quat_apply

    base = env.unwrapped
    robot = base.scene["robot"]
    wrist = robot.body_names.index("wrist_3_link")
    num_envs = base.num_envs
    device = base.device
    step_dt = base.step_dt
    offset = torch.tensor(tcp_offset, device=device).repeat(num_envs, 1)
    command = base.command_manager._terms["ee_pose"]

    error_steps: list[torch.Tensor] = []
    cross_steps: list[torch.Tensor] = []
    along_steps: list[torch.Tensor] = []
    captured_steps: list[torch.Tensor] = []
    hull_steps: list[torch.Tensor] = []
    tcp_steps: list[torch.Tensor] = []
    capture_times: list[float] = []
    prior_tcp = None
    captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episodes_started = 0
    episodes_captured = 0

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            episode_step = base.episode_length_buf.clone()
            fresh = episode_step == 0
            if bool(fresh.any()):
                episodes_started += int(fresh.sum())
                captured[fresh] = False

            tcp = robot.data.body_pos_w[:, wrist] + quat_apply(
                robot.data.body_quat_w[:, wrist], offset
            )
            target = base.scene.env_origins + command.pose_command_b[:, :3]
            error = torch.linalg.norm(tcp - target, dim=1)

            # Geometric line-following error: perpendicular distance to the infinite
            # line (cross-track) and lag along it (along-track). The scalar `error`
            # above mixes the two.
            center = base.scene.env_origins + command.line_center_e
            direction = getattr(command, "line_direction_e", None)
            if direction is None:
                direction = torch.zeros_like(center)
                direction[:, 1] = 1.0
            to_tcp = tcp - center
            to_target = target - center
            along_tcp = (to_tcp * direction).sum(dim=1)
            cross = torch.linalg.norm(to_tcp - along_tcp[:, None] * direction, dim=1)
            along_error = (along_tcp - (to_target * direction).sum(dim=1)).abs()

            just_captured = (~captured) & (error <= capture_threshold)
            if bool(just_captured.any()):
                episodes_captured += int(just_captured.sum())
                capture_times.extend((episode_step[just_captured].float() * step_dt).cpu().tolist())
            captured |= error <= capture_threshold

            error_steps.append(error)
            cross_steps.append(cross)
            along_steps.append(along_error)
            captured_steps.append(captured.clone())
            # How far the hull itself has wandered. On a fixed-base task this is zero by
            # construction; on a free-floating one it is the disturbance the arm is
            # rejecting, and reporting the tracking error without it would leave the
            # difficulty of the condition unstated.
            origin = robot.data.default_root_state[:, :3] + base.scene.env_origins
            hull_steps.append(torch.linalg.norm(robot.data.root_pos_w - origin, dim=1))
            if prior_tcp is not None:
                tcp_steps.append(torch.linalg.norm(tcp - prior_tcp, dim=1))
            prior_tcp = tcp.clone()

            obs, _, dones, _ = env.step(policy(obs))
            if bool(dones.any()) and hasattr(policy, "reset"):
                policy.reset(dones)

    errors = torch.stack(error_steps)
    cross = torch.stack(cross_steps)
    along = torch.stack(along_steps)
    tracking = torch.stack(captured_steps)
    hull = torch.stack(hull_steps)
    capture = torch.tensor(capture_times, dtype=torch.float64)

    result = {
        "mean_tcp_step_m": float(torch.stack(tcp_steps).mean()) if tcp_steps else float("nan"),
        "hull_excursion_mean_m": float(hull.mean()),
        "hull_excursion_max_m": float(hull.max()),
        # Fraction of rollout spent transiting to the line rather than tracking it.
        "tracking_sample_fraction": float(tracking.double().mean()),
        "episodes_started": episodes_started,
        "episodes_captured": episodes_captured,
        "capture_rate": episodes_captured / episodes_started if episodes_started else float("nan"),
        "capture_time_mean_s": float(capture.mean()) if capture.numel() else float("nan"),
        "capture_time_p95_s": float(capture.quantile(0.95)) if capture.numel() else float("nan"),
    }
    result.update(stats(errors.flatten(), "all_"))
    result.update(stats(errors[tracking], "track_"))
    result.update(stats(cross[tracking], "track_cross_"))
    result.update(stats(along[tracking], "track_along_"))
    return result
