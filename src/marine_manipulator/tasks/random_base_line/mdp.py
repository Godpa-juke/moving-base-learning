from __future__ import annotations

import math

import torch
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
)

from marine_manipulator.motion_config import AXES, MOTION_RANGES

from marine_manipulator.calibration import (  # noqa: F401  (re-exported)
    PRECISION_FAR_START_JOINT_POS,
    PRECISION_FAR_START_TCP_E,
    PRECISION_START_HOLD_S,
    PRECISION_START_JOINT_POS,
    PRECISION_START_TCP_E,
    REALISTIC_ACTION_SCALE_RAD,
    REALISTIC_JOINT_SPEED_LIMIT_RAD_S,
    TCP_OFFSET,
)

_STATE_ATTR = "_marine_random_base_motion"


class WorldHorizontalLineCommand(UniformPoseCommand):
    """Horizontal sinusoidal target whose pose is fixed in world axes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.body_offset_b = torch.tensor(cfg.body_offset, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )
        self.line_center_e = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.line_amplitude = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.line_period = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.line_phase = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)
        self.line_center_e[env_ids] = self.pose_command_b[env_ids, :3]
        random = torch.empty(len(env_ids), dtype=torch.float32, device=self.device)
        self.line_amplitude[env_ids] = random.uniform_(*self.cfg.amplitude_range)
        self.line_period[env_ids] = random.uniform_(*self.cfg.period_range)
        self.line_phase[env_ids] = random.uniform_(0.0, 2.0 * math.pi)

    def _update_command(self) -> None:
        elapsed = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
        phase = 2.0 * math.pi * elapsed / self.line_period + self.line_phase
        offset = self.line_amplitude * torch.sin(phase)
        self.pose_command_b[:, :3] = self.line_center_e
        self.pose_command_b[:, 1] += offset
        # UniformPoseCommand updates metrics before commands each manager step.
        # Refresh the world target here so rewards/observations see the current
        # line point rather than the previous step's point.
        self.pose_command_w[:, :3] = self._env.scene.env_origins + self.pose_command_b[:, :3]
        self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:]

    def _update_metrics(self) -> None:
        self.pose_command_w[:, :3] = self._env.scene.env_origins + self.pose_command_b[:, :3]
        self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:]
        position = self.robot.data.body_pos_w[:, self.body_idx]
        quaternion = self.robot.data.body_quat_w[:, self.body_idx]
        tcp = position + quat_apply(quaternion, self.body_offset_b)
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:], tcp, quaternion
        )
        self.metrics["position_error"] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.linalg.norm(rot_error, dim=-1)


class ReachableRandomLineCommand(WorldHorizontalLineCommand):
    """Sample an independently located and oriented reachable horizontal line per episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.line_direction_e = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _resample_command(self, env_ids) -> None:
        WorldHorizontalLineCommand._resample_command(self, env_ids)
        count = len(env_ids)
        self.line_center_e[env_ids, 0] = torch.empty(
            count, dtype=torch.float32, device=self.device
        ).uniform_(*self.cfg.ranges.pos_x)
        self.line_center_e[env_ids, 1] = torch.empty(
            count, dtype=torch.float32, device=self.device
        ).uniform_(*self.cfg.ranges.pos_y)
        self.line_center_e[env_ids, 2] = torch.empty(
            count, dtype=torch.float32, device=self.device
        ).uniform_(*self.cfg.ranges.pos_z)
        # Keep every rendered bar on the requested world-axis direction.
        direction = torch.tensor(self.cfg.line_direction_e, dtype=torch.float32, device=self.device)
        direction = direction / torch.linalg.norm(direction)
        self.line_direction_e[env_ids] = direction
        self.line_phase[env_ids] = 0.0
        self.pose_command_b[env_ids, :3] = (
            self.line_center_e[env_ids]
            - self.line_direction_e[env_ids] * self.line_amplitude[env_ids, None]
        )
        self.pose_command_w[env_ids, :3] = (
            self._env.scene.env_origins[env_ids] + self.pose_command_b[env_ids, :3]
        )
        self.pose_command_w[env_ids, 3:] = self.pose_command_b[env_ids, 3:]

    def _update_command(self) -> None:
        elapsed = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
        moving_time = torch.clamp(elapsed - self.cfg.hold_duration_s, min=0.0)
        offset = -self.line_amplitude * torch.cos(
            math.pi * moving_time / self.cfg.traverse_duration_s
        )
        self.pose_command_b[:, :3] = self.line_center_e + self.line_direction_e * offset[:, None]
        self.pose_command_w[:, :3] = self._env.scene.env_origins + self.pose_command_b[:, :3]
        self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:]


class PrecisionStartLineCommand(WorldHorizontalLineCommand):
    """Hold at the -Y endpoint, then cosine-ease between line endpoints."""

    def _resample_command(self, env_ids) -> None:
        super()._resample_command(env_ids)
        self.line_phase[env_ids] = 0.0
        self.pose_command_b[env_ids, :3] = self.line_center_e[env_ids]
        self.pose_command_b[env_ids, 1] -= self.line_amplitude[env_ids]
        self.pose_command_w[env_ids, :3] = (
            self._env.scene.env_origins[env_ids] + self.pose_command_b[env_ids, :3]
        )
        self.pose_command_w[env_ids, 3:] = self.pose_command_b[env_ids, 3:]

    def _update_command(self) -> None:
        elapsed = self._env.episode_length_buf.to(torch.float32) * self._env.step_dt
        moving_time = torch.clamp(elapsed - self.cfg.hold_duration_s, min=0.0)
        offset = -self.line_amplitude * torch.cos(
            math.pi * moving_time / self.cfg.traverse_duration_s
        )
        self.pose_command_b[:, :3] = self.line_center_e
        self.pose_command_b[:, 1] += offset
        self.pose_command_w[:, :3] = self._env.scene.env_origins + self.pose_command_b[:, :3]
        self.pose_command_w[:, 3:] = self.pose_command_b[:, 3:]


def reset_joints_to_start_pose(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    positions = torch.tensor(PRECISION_START_JOINT_POS, device=env.device).repeat(len(env_ids), 1)
    velocities = torch.zeros_like(positions)
    asset.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)


def reset_joints_to_far_start_pose(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    positions = torch.tensor(PRECISION_FAR_START_JOINT_POS, device=env.device).repeat(len(env_ids), 1)
    velocities = torch.zeros_like(positions)
    asset.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)


def position_success(
    env, threshold: float, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    return (position_error(env, command_name, asset_cfg, body_offset) <= threshold).to(torch.float32)


def _range_tensors(device):
    amp_low = torch.tensor([MOTION_RANGES[a].amplitude[0] for a in AXES], device=device)
    amp_high = torch.tensor([MOTION_RANGES[a].amplitude[1] for a in AXES], device=device)
    freq_low = torch.tensor([MOTION_RANGES[a].frequency[0] for a in AXES], device=device)
    freq_high = torch.tensor([MOTION_RANGES[a].frequency[1] for a in AXES], device=device)
    return amp_low, amp_high, freq_low, freq_high


def rate_limit_joint_action(
    previous: torch.Tensor,
    requested: torch.Tensor,
    max_speed_rad_s: torch.Tensor,
    step_dt: float,
    action_scale: float,
) -> torch.Tensor:
    """Clamp normalized joint-position action changes to a physical target speed."""
    max_delta = max_speed_rad_s.to(device=requested.device, dtype=requested.dtype) * (
        step_dt / action_scale
    )
    return previous + torch.clamp(requested - previous, min=-max_delta, max=max_delta)


_WAVE_SCALE_ATTR = "_marine_wave_scale"


def set_wave_scale(env, factor: float) -> None:
    """Scale every episode's disturbance amplitude, for probes that need it silenced.

    Applied where the amplitude is drawn rather than where it is used, so that
    everything downstream - the prescribed base motion of the fixed-base tasks, the
    wave force of the free-floating one, and the wave-consistent initial hull velocity
    written by :func:`reset_uvms_root` - all see the same silenced disturbance. Scaling
    at the point of use would leave the hull launched at a velocity for a wave that is
    no longer there.
    """
    setattr(env, _WAVE_SCALE_ATTR, float(factor))


_WAVE_FREQUENCY_SCALE_ATTR = "_marine_wave_frequency_scale"


_CONDITIONS_PINNED_ATTR = "_marine_actuation_conditions_pinned"


def pin_actuation_conditions(env, pinned: bool = True) -> None:
    """Make :func:`sample_actuation_conditions` a no-op.

    An evaluation pins the delay and the disturbance speed to the cell being measured,
    but the reset event that randomises them for training fires afterwards and would
    silently overwrite the pin — which is how a zero residual came to read 6.29 mm
    against baseline A's 3.68 mm on what should have been the identical condition. The
    pin therefore has to take precedence explicitly rather than by ordering.
    """
    setattr(env, _CONDITIONS_PINNED_ATTR, bool(pinned))


def set_wave_frequency_scale(env, factor: float) -> None:
    """Multiply every episode's disturbance frequency.

    The disturbance in this study runs at 0.06-0.3 Hz against a 30 Hz control loop, so
    there are 100-500 control steps per period and feedback rejects it almost entirely
    (``docs/FINDINGS.md`` 16). Herland & Bach's vessel trajectories, after their
    ``lambda = 10`` scaling, sit at 0.32-0.53 Hz. This exists to sweep the ratio rather
    than assume it, since it is one of four differences between their positive result
    and this study's negative one.
    """
    setattr(env, _WAVE_FREQUENCY_SCALE_ATTR, float(factor))
    steps, _ = getattr(env, "_marine_actuation_conditions", (0, 1.0))
    setattr(env, "_marine_actuation_conditions", (steps, float(factor)))


_ACTUATION_DELAY_ATTR = "_marine_actuation_delay_steps"
_ACTUATION_BUFFER_ATTR = "_marine_actuation_buffer"


def set_actuation_delay(env, steps: int) -> None:
    """Delay the command reaching the arm by whole control steps.

    Distinct from the sensor delay above, and the distinction is the point. A
    *measurement* delay makes the controller act on stale information about a signal
    that is analytically predictable; an *actuation* delay sits inside the feedback loop
    and costs phase margin directly. The previous round swept measurement delay to
    333 ms and found the analytic controller flat, while Herland & Bach report their IK
    baseline degrading roughly fourfold on 40 ms of actuation delay. This term exists to
    test whether that difference is the explanation.
    """
    setattr(env, _ACTUATION_DELAY_ATTR, int(steps))
    if hasattr(env, _ACTUATION_BUFFER_ATTR):
        delattr(env, _ACTUATION_BUFFER_ATTR)
    _, scale = getattr(env, "_marine_actuation_conditions", (0, 1.0))
    setattr(env, "_marine_actuation_conditions", (int(steps), scale))


def sample_actuation_conditions(
    env,
    env_ids: torch.Tensor | None,
    delay_steps_range: tuple[int, int],
    frequency_scale_range: tuple[float, float],
) -> None:
    """Draw the actuation delay and disturbance speed for the coming episodes.

    Both are environment-wide rather than per-environment: the delay is implemented as a
    FIFO shared by the whole batch, and a per-environment delay would need a separate
    buffer depth per row. Drawing once per reset wave is enough to cover the range over
    training, and the evaluation pins both anyway.
    """
    if getattr(env, _CONDITIONS_PINNED_ATTR, False):
        return
    low, high = delay_steps_range
    steps = int(torch.randint(int(low), int(high) + 1, (1,)).item())
    scale_low, scale_high = frequency_scale_range
    scale = float(torch.empty(1).uniform_(scale_low, scale_high).item())
    set_actuation_delay(env, steps)
    set_wave_frequency_scale(env, scale)
    # Cached for the observation term, which must report what was actually set.
    setattr(env, "_marine_actuation_conditions", (steps, scale))
    # The disturbance for these environments was drawn by `sample_base_motion` earlier in
    # this same reset, at the previous episode's frequency. Redrawing it here is what
    # keeps the motion consistent with the number the observation is about to report.
    sample_base_motion(env, env_ids)


def normalized_actuation_conditions(env) -> torch.Tensor:
    """The delay and disturbance speed, as the policy sees them, ``(num_envs, 2)``.

    A single frame cannot reveal how stale the commands are, and the alternative -
    training one agent per delay, as Herland & Bach do - answers a different question.
    Telling the policy is the honest way to keep one agent across the range, and the
    analytic controller is handed nothing by it because it has no use for the number.
    """
    steps, scale = getattr(env, "_marine_actuation_conditions", (0, 1.0))
    values = torch.tensor(
        [float(steps) / 3.0, (float(scale) - 1.0) / 3.0], device=env.device, dtype=torch.float32
    )
    return values.unsqueeze(0).expand(env.num_envs, 2)


def delay_actuation(env, command: torch.Tensor) -> torch.Tensor:
    """Return the command the arm actually receives this step.

    A FIFO of past commands, one slot per control step. The buffer is primed with the
    first command rather than with zeros, so an episode does not open by driving the arm
    to a posture nobody asked for.
    """
    steps = int(getattr(env, _ACTUATION_DELAY_ATTR, 0))
    if steps <= 0:
        return command
    buffer = getattr(env, _ACTUATION_BUFFER_ATTR, None)
    if buffer is None or buffer.shape[0] != steps or buffer.shape[1] != command.shape[0]:
        buffer = command.unsqueeze(0).repeat(steps, 1, 1)
        setattr(env, _ACTUATION_BUFFER_ATTR, buffer)
    applied = buffer[0].clone()
    buffer[:-1] = buffer[1:].clone()
    buffer[-1] = command
    return applied


def sample_base_motion(env, env_ids: torch.Tensor | None) -> None:
    """Sample independent amplitude, frequency, and phase for all 6 axes per reset."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    state = getattr(env, _STATE_ATTR, None)
    if state is None or state["amplitude"].shape[0] != env.num_envs:
        state = {
            "amplitude": torch.zeros((env.num_envs, 6), device=env.device),
            "frequency": torch.zeros((env.num_envs, 6), device=env.device),
            "phase": torch.zeros((env.num_envs, 6), device=env.device),
        }
        setattr(env, _STATE_ATTR, state)
    amp_low, amp_high, freq_low, freq_high = _range_tensors(env.device)
    n = int(env_ids.numel())
    state["amplitude"][env_ids] = (amp_low + torch.rand((n, 6), device=env.device) * (amp_high - amp_low)) * float(
        getattr(env, _WAVE_SCALE_ATTR, 1.0)
    )
    state["frequency"][env_ids] = (
        freq_low + torch.rand((n, 6), device=env.device) * (freq_high - freq_low)
    ) * float(getattr(env, _WAVE_FREQUENCY_SCALE_ATTR, 1.0))
    state["phase"][env_ids] = torch.rand((n, 6), device=env.device) * (2.0 * math.pi)


def sample_precision_base_motion(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Sample motion and reset the externally driven base to its neutral pose."""
    sample_base_motion(env, env_ids)
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    default = asset.data.default_root_state[env_ids].clone()
    root_pose = default[:, :7].clone()
    root_pose[:, :3] = default[:, :3] + env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(default[:, 7:13], env_ids=env_ids)
    asset.update(0.0)
    setattr(env, "_precision_reset_root_pose", root_pose.detach().clone())


def _state(env):
    state = getattr(env, _STATE_ATTR, None)
    if state is None:
        sample_base_motion(env, None)
        state = getattr(env, _STATE_ATTR)
    return state


def base_pose(env) -> torch.Tensor:
    state = _state(env)
    t = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    return state["amplitude"] * torch.sin(2.0 * math.pi * state["frequency"] * t + state["phase"])


def base_velocity(env) -> torch.Tensor:
    state = _state(env)
    t = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    omega = 2.0 * math.pi * state["frequency"]
    return state["amplitude"] * omega * torch.cos(omega * t + state["phase"])


def neutral_start_base_pose(env) -> torch.Tensor:
    """Random 6-DoF motion that is exactly neutral at episode time zero."""
    state = _state(env)
    elapsed = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    t = torch.clamp(elapsed - PRECISION_START_HOLD_S, min=0.0)
    omega_t = 2.0 * math.pi * state["frequency"] * t
    return 0.5 * state["amplitude"] * (
        torch.sin(omega_t + state["phase"]) - torch.sin(state["phase"])
    )


def neutral_start_base_velocity(env) -> torch.Tensor:
    state = _state(env)
    elapsed = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    t = torch.clamp(elapsed - PRECISION_START_HOLD_S, min=0.0)
    omega = 2.0 * math.pi * state["frequency"]
    active = (elapsed > PRECISION_START_HOLD_S).to(torch.float32)
    return active * 0.5 * state["amplitude"] * omega * torch.cos(omega * t + state["phase"])


def normalized_neutral_start_base_pose(env) -> torch.Tensor:
    _, amp_high, _, _ = _range_tensors(env.device)
    return torch.clamp(neutral_start_base_pose(env) / amp_high, -1.0, 1.0)


def immediate_base_pose(env) -> torch.Tensor:
    """Random motion starts at neutral position but has nonzero velocity at t=0."""
    state = _state(env)
    t = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    omega_t = 2.0 * math.pi * state["frequency"] * t
    return 0.5 * state["amplitude"] * (
        torch.sin(omega_t + state["phase"]) - torch.sin(state["phase"])
    )


def immediate_base_velocity(env) -> torch.Tensor:
    state = _state(env)
    t = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    omega = 2.0 * math.pi * state["frequency"]
    return 0.5 * state["amplitude"] * omega * torch.cos(omega * t + state["phase"])


def normalized_immediate_base_pose(env) -> torch.Tensor:
    _, amp_high, _, _ = _range_tensors(env.device)
    return torch.clamp(immediate_base_pose(env) / amp_high, -1.0, 1.0)


def normalized_immediate_base_velocity(env) -> torch.Tensor:
    _, amp_high, _, freq_high = _range_tensors(env.device)
    scale = amp_high * 2.0 * math.pi * freq_high
    return torch.clamp(immediate_base_velocity(env) / scale, -1.0, 1.0)


def normalized_neutral_start_base_velocity(env) -> torch.Tensor:
    _, amp_high, _, freq_high = _range_tensors(env.device)
    scale = amp_high * 2.0 * math.pi * freq_high
    return torch.clamp(neutral_start_base_velocity(env) / scale, -1.0, 1.0)


def normalized_base_pose(env) -> torch.Tensor:
    _, amp_high, _, _ = _range_tensors(env.device)
    return torch.clamp(base_pose(env) / amp_high, -1.0, 1.0)


def normalized_base_velocity(env) -> torch.Tensor:
    _, amp_high, _, freq_high = _range_tensors(env.device)
    scale = amp_high * 2.0 * math.pi * freq_high
    return torch.clamp(base_velocity(env) / scale, -1.0, 1.0)


def apply_base_motion(env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    default = asset.data.default_root_state[env_ids].clone()
    pose = base_pose(env)[env_ids]
    velocity = base_velocity(env)[env_ids]
    root_pose = default[:, :7].clone()
    root_pose[:, :3] = default[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
    delta_q = quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    root_pose[:, 3:7] = quat_mul(default[:, 3:7], delta_q)
    root_velocity = default[:, 7:13].clone()
    root_velocity[:, :6] = velocity
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def apply_neutral_start_base_motion(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    default = asset.data.default_root_state[env_ids].clone()
    pose = neutral_start_base_pose(env)[env_ids]
    velocity = neutral_start_base_velocity(env)[env_ids]
    root_pose = default[:, :7].clone()
    root_pose[:, :3] = default[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
    delta_q = quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    root_pose[:, 3:7] = quat_mul(default[:, 3:7], delta_q)
    root_velocity = default[:, 7:13].clone()
    root_velocity[:, :6] = velocity
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def apply_immediate_base_motion(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    default = asset.data.default_root_state[env_ids].clone()
    pose = immediate_base_pose(env)[env_ids]
    velocity = immediate_base_velocity(env)[env_ids]
    root_pose = default[:, :7].clone()
    root_pose[:, :3] = default[:, :3] + env.scene.env_origins[env_ids] + pose[:, :3]
    delta_q = quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5])
    root_pose[:, 3:7] = quat_mul(default[:, 3:7], delta_q)
    root_velocity = default[:, 7:13].clone()
    root_velocity[:, :6] = velocity
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)


def _tcp(env, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET):
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    offset = torch.tensor(body_offset, device=env.device).repeat(env.num_envs, 1)
    return asset.data.body_pos_w[:, body_id] + quat_apply(asset.data.body_quat_w[:, body_id], offset)


def tcp_position_w(env, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET) -> torch.Tensor:
    return _tcp(env, asset_cfg, body_offset)


def tcp_speed_limit_penalty(
    env, max_speed_m_s: float, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    offset = torch.tensor(body_offset, device=env.device).repeat(env.num_envs, 1)
    rotated_offset = quat_apply(asset.data.body_quat_w[:, body_id], offset)
    tcp_velocity = asset.data.body_lin_vel_w[:, body_id] + torch.cross(
        asset.data.body_ang_vel_w[:, body_id], rotated_offset, dim=1
    )
    excess = torch.relu(torch.linalg.norm(tcp_velocity, dim=1) - max_speed_m_s)
    return excess.square()


def target_error_w(
    env, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    """World-frame vector from the true cylinder-tip TCP to the rendered command target."""
    return _target(env, command_name) - _tcp(env, asset_cfg, body_offset)


def line_endpoints_w(env, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager._terms[command_name]
    center = env.scene.env_origins + command.line_center_e
    direction = getattr(command, "line_direction_e", None)
    if direction is None:
        direction = torch.zeros_like(center)
        direction[:, 1] = 1.0
    half_line = direction * command.line_amplitude[:, None]
    return center - half_line, center + half_line


def _target(env, command_name: str) -> torch.Tensor:
    return env.command_manager._terms[command_name].pose_command_w[:, :3]


def position_error(env, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET) -> torch.Tensor:
    return torch.linalg.norm(_tcp(env, asset_cfg, body_offset) - _target(env, command_name), dim=1)


def position_error_tanh(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    return 1.0 - torch.tanh(position_error(env, command_name, asset_cfg, body_offset) / std)


def orientation_error(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    target = env.command_manager._terms[command_name].pose_command_w[:, 3:7]
    return quat_error_magnitude(asset.data.body_quat_w[:, body_id], target)


# --------------------------------------------------------------------------------------
# IK-seeded start, tool-axis orientation, and raw-action accounting.
#
# The rate limiter in the env classes rewrites the policy's output before the action
# manager ever sees it, so `action_l2` / `action_rate_l2` describe the limited signal
# and leave the raw output completely unpenalised. The actor mean is then free to
# drift outward until the limiter saturates and control degenerates into a bang-bang
# ramp, which is what makes every long run lose precision late in training. The terms
# below read the raw output the env stashes so the penalty lands where it belongs.
# --------------------------------------------------------------------------------------

_RAW_ACTION_ATTR = "_marine_raw_policy_action"
_PREV_RAW_ACTION_ATTR = "_marine_previous_raw_policy_action"

#: World-frame direction the cylinder axis should point (straight down onto the line).
TOOL_AXIS_TARGET_W = (0.0, 0.0, -1.0)


def raw_action_l2(env) -> torch.Tensor:
    """Squared magnitude of the policy's own output, before rate limiting."""
    raw = getattr(env, _RAW_ACTION_ATTR, None)
    if raw is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(raw.square(), dim=1)


def raw_action_rate_l2(env) -> torch.Tensor:
    """Squared step-to-step change of the policy's own output, before rate limiting."""
    raw = getattr(env, _RAW_ACTION_ATTR, None)
    previous = getattr(env, _PREV_RAW_ACTION_ATTR, None)
    if raw is None or previous is None:
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum((raw - previous).square(), dim=1)


def tool_axis_world(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unit cylinder axis in world coordinates, ``(num_envs, 3)``."""
    asset = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    local_axis = torch.zeros((env.num_envs, 3), device=env.device)
    local_axis[:, 2] = 1.0
    return quat_apply(asset.data.body_quat_w[:, body_id], local_axis)


def tool_axis_angle(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angle in radians between the cylinder axis and straight down, ``(num_envs,)``."""
    target = torch.tensor(TOOL_AXIS_TARGET_W, device=env.device).expand(env.num_envs, 3)
    cosine = (tool_axis_world(env, asset_cfg) * target).sum(dim=1)
    return torch.acos(cosine.clamp(-1.0, 1.0))


def tool_axis_alignment(env, std: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bounded reward for holding the cylinder perpendicular to the horizontal line.

    Without this the tip can be placed by pivoting the 0.12 m tool about the wrist,
    a degenerate solution that traces the line with the cylinder lying almost flat.
    """
    return 1.0 - torch.tanh(tool_axis_angle(env, asset_cfg) / std)


def reset_joints_to_ik_line_start(
    env,
    env_ids: torch.Tensor,
    command_name: str = "ee_pose",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_offset=TCP_OFFSET,
    iterations: int = 80,
) -> None:
    """Place the arm on the line's start point with the cylinder vertical.

    Must run from ``step()`` rather than a reset event: ``ManagerBasedRLEnv._reset_idx``
    applies events *before* ``command_manager.reset``, so an event would seed against
    the previous episode's line.
    """
    from marine_manipulator import calibration, ur3_kin

    asset = env.scene[asset_cfg.name]
    command = env.command_manager._terms[command_name]
    target_w = env.scene.env_origins[env_ids] + command.pose_command_b[env_ids, :3]

    root_pos = asset.data.root_pos_w[env_ids]
    root_quat = asset.data.root_quat_w[env_ids]
    target_root = quat_apply_inverse(root_quat, target_w - root_pos)
    axis_w = torch.tensor(TOOL_AXIS_TARGET_W, device=env.device).expand(len(env_ids), 3)
    axis_root = quat_apply_inverse(root_quat, axis_w)

    seed = torch.tensor(calibration.VERTICAL_TOOL_IK_SEED, device=env.device).repeat(len(env_ids), 1)
    reference = torch.tensor(calibration.BOX_CENTER_JOINT_POS, device=env.device).repeat(
        len(env_ids), 1
    )
    positions, _ = ur3_kin.inverse_kinematics(
        target_root,
        ur3_kin.rotation_from_tool_axis(axis_root),
        seed,
        body_offset[2],
        iterations=iterations,
    )
    positions = ur3_kin.wrap_to_reference(positions, reference)
    asset.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)
    asset.update(0.0)
    _recenter_action_offset(env, env_ids, positions)


def _recenter_action_offset(env, env_ids: torch.Tensor, positions: torch.Tensor) -> None:
    """Re-origin the joint-position action space on this episode's start pose.

    ``JointPositionAction`` with ``use_default_offset`` clones the articulation's
    default joint positions once at construction, so a zero action would drive the
    arm back to that single posture no matter where IK just placed it. Writing the
    per-environment offset makes a zero action mean 'hold the line start', which is
    both the correct prior and a far better conditioned parametrisation.
    """
    from marine_manipulator import ur3_kin

    term = env.action_manager._terms.get("arm_action")
    if term is None or not isinstance(getattr(term, "_offset", None), torch.Tensor):
        return
    names = tuple(getattr(term, "_joint_names", ur3_kin.JOINT_NAMES))
    if names != ur3_kin.JOINT_NAMES:
        raise RuntimeError(
            f"arm_action joint order {names} does not match the kinematics order "
            f"{ur3_kin.JOINT_NAMES}; the IK solution would be applied to the wrong joints"
        )
    term._offset[env_ids] = positions.to(term._offset.dtype)


def reset_joints_to_box_center(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Reset to the workspace-centre posture, the origin of the action space.

    ``step()`` refines this to the exact IK solution for the sampled line, but the
    observation returned by ``_reset_idx`` is computed before that, so the arm must
    already be somewhere sensible.
    """
    from marine_manipulator import calibration

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    positions = torch.tensor(calibration.BOX_CENTER_JOINT_POS, device=env.device).repeat(
        len(env_ids), 1
    )
    asset.write_joint_state_to_sim(positions, torch.zeros_like(positions), env_ids=env_ids)


# --------------------------------------------------------------------------------------
# Degraded base measurement.
#
# The base still *moves* exactly as sampled - `apply_immediate_base_motion` is
# untouched. Only what the controllers get to *measure* is degraded. Because the
# motion is analytic, a delay is evaluated by shifting the time argument rather than
# buffering history, which is exact and costs nothing.
#
# The measurement is cached within a control step because several readers share it -
# the pose and rate observation terms, and the model-based controller - and they must
# agree on one noise draw rather than each taking an independent one. The cache is
# invalidated at the top of every env step; a controller acting before the step and
# the observation computed after it are different instants and correctly differ.
# --------------------------------------------------------------------------------------

_SENSOR_ATTR = "_marine_sensor_degradation"
_SENSOR_CACHE_ATTR = "_marine_sensor_measurement_cache"


def _ensure_sensor_state(env):
    """Zero-initialised degradation, i.e. a perfect sensor unless told otherwise.

    Defaulting to zeros rather than a random draw matters: tasks that never configure
    the reset event must keep reading the true base state, so that the same
    measurement helper can serve both the clean and the degraded task.
    """
    state = getattr(env, _SENSOR_ATTR, None)
    if state is None or state["delay_s"].shape[0] != env.num_envs:
        state = {
            "delay_s": torch.zeros(env.num_envs, device=env.device),
            "position_noise_m": torch.zeros(env.num_envs, device=env.device),
            "rotation_noise_rad": torch.zeros(env.num_envs, device=env.device),
        }
        setattr(env, _SENSOR_ATTR, state)
    return state


def sample_sensor_degradation(
    env,
    env_ids: torch.Tensor | None,
    delay_range_s: tuple[float, float],
    position_noise_range_m: tuple[float, float],
    rotation_noise_range_rad: tuple[float, float],
) -> None:
    """Draw a per-environment measurement delay and noise level at reset.

    Ranges are event parameters rather than module constants so that an evaluation
    sweep can pin them to a single value by passing a degenerate range.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    state = _ensure_sensor_state(env)
    count = int(env_ids.numel())
    for key, (low, high) in (
        ("delay_s", delay_range_s),
        ("position_noise_m", position_noise_range_m),
        ("rotation_noise_rad", rotation_noise_range_rad),
    ):
        state[key][env_ids] = low + torch.rand(count, device=env.device) * (high - low)


def set_sensor_degradation(
    env,
    delay_s: float | None = None,
    position_noise_m: float | None = None,
    rotation_noise_rad: float | None = None,
) -> None:
    """Pin the degradation to fixed values, for sweeping at evaluation time."""
    state = _ensure_sensor_state(env)
    for key, value in (
        ("delay_s", delay_s),
        ("position_noise_m", position_noise_m),
        ("rotation_noise_rad", rotation_noise_rad),
    ):
        if value is not None:
            state[key][:] = value


def clear_sensor_cache(env) -> None:
    """Drop the per-step measurement so the next read draws fresh noise."""
    if hasattr(env, _SENSOR_CACHE_ATTR):
        delattr(env, _SENSOR_CACHE_ATTR)


def measured_base_state(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Delayed, noisy base offset and rate, ``(num_envs, 6)`` each."""
    cached = getattr(env, _SENSOR_CACHE_ATTR, None)
    if cached is not None:
        return cached

    motion = _state(env)
    sensor = _ensure_sensor_state(env)
    elapsed = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    # A delay is a shift of the analytic time argument, clamped so the measurement
    # never predates the episode.
    delayed = torch.clamp(elapsed - sensor["delay_s"].unsqueeze(1), min=0.0)
    omega = 2.0 * math.pi * motion["frequency"]
    offset = 0.5 * motion["amplitude"] * (
        torch.sin(omega * delayed + motion["phase"]) - torch.sin(motion["phase"])
    )
    rate = 0.5 * motion["amplitude"] * omega * torch.cos(omega * delayed + motion["phase"])

    noise_scale = torch.cat(
        (
            sensor["position_noise_m"].unsqueeze(1).expand(-1, 3),
            sensor["rotation_noise_rad"].unsqueeze(1).expand(-1, 3),
        ),
        dim=1,
    )
    offset = offset + torch.randn_like(offset) * noise_scale
    # Differentiating a noisy signal amplifies it; a real unit reports rate from a
    # separate channel, so scale the rate noise by the disturbance bandwidth instead.
    rate = rate + torch.randn_like(rate) * noise_scale * omega

    measurement = (offset, rate)
    setattr(env, _SENSOR_CACHE_ATTR, measurement)
    return measurement


def normalized_measured_base_pose(env) -> torch.Tensor:
    _, amp_high, _, _ = _range_tensors(env.device)
    return torch.clamp(measured_base_state(env)[0] / amp_high, -1.0, 1.0)


def normalized_measured_base_velocity(env) -> torch.Tensor:
    _, amp_high, _, freq_high = _range_tensors(env.device)
    scale = amp_high * 2.0 * math.pi * freq_high
    return torch.clamp(measured_base_state(env)[1] / scale, -1.0, 1.0)


def measured_root_pose(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Root pose a controller would read from the degraded sensor, ``(pos, quat)``.

    Mirrors :func:`apply_immediate_base_motion` so that at zero delay and zero noise
    it reproduces the true root pose exactly.
    """
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_root_state
    offset = measured_base_state(env)[0]
    position = default[:, :3] + env.scene.env_origins + offset[:, :3]
    delta = quat_from_euler_xyz(offset[:, 3], offset[:, 4], offset[:, 5])
    return position, quat_mul(default[:, 3:7], delta)


# --------------------------------------------------------------------------------------
# Degraded seam measurement.
#
# `target_error_w` hands the policy the exact vector from the true TCP to the true
# target. That is a stronger signal than anything else in the observation: with it the
# policy can close the loop on its own error and never needs to know the base is moving
# at all, which is why degrading only the base measurement changed nothing. A real seam
# tracker reports that offset late and noisily, so it is degraded on the same terms.
#
# Unlike the base motion this cannot be delayed analytically - it depends on where the
# arm actually is - so past values are buffered. The buffer is advanced once per control
# step, guarded by the same per-step cache as the base measurement.
# --------------------------------------------------------------------------------------

_SEAM_BUFFER_ATTR = "_marine_seam_error_buffer"
_SEAM_CURSOR_ATTR = "_marine_seam_error_cursor"
_SEAM_CACHE_ATTR = "_marine_seam_measurement_cache"


def clear_seam_cache(env) -> None:
    if hasattr(env, _SEAM_CACHE_ATTR):
        delattr(env, _SEAM_CACHE_ATTR)


def reset_seam_buffer(env, env_ids: torch.Tensor | None) -> None:
    """Refill an environment's history with its current error, so a fresh episode does
    not read a delayed measurement left over from the previous one."""
    buffer = getattr(env, _SEAM_BUFFER_ATTR, None)
    if buffer is None:
        return
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    buffer[env_ids.to(env.device, torch.long)] = 0.0


def measured_target_error_w(
    env, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    """Seam offset as a tracker would report it: delayed by the sensor lag and noisy."""
    from marine_manipulator import calibration

    cached = getattr(env, _SEAM_CACHE_ATTR, None)
    if cached is not None:
        return cached

    true_error = _target(env, command_name) - _tcp(env, asset_cfg, body_offset)
    depth = calibration.SEAM_BUFFER_STEPS
    buffer = getattr(env, _SEAM_BUFFER_ATTR, None)
    if buffer is None or buffer.shape[0] != env.num_envs:
        buffer = true_error.unsqueeze(1).repeat(1, depth, 1)
        setattr(env, _SEAM_BUFFER_ATTR, buffer)
        setattr(env, _SEAM_CURSOR_ATTR, 0)
    cursor = (int(getattr(env, _SEAM_CURSOR_ATTR, 0)) + 1) % depth
    setattr(env, _SEAM_CURSOR_ATTR, cursor)
    buffer[:, cursor] = true_error

    sensor = _ensure_sensor_state(env)
    lag = torch.round(sensor["delay_s"] / env.step_dt).long().clamp_(0, depth - 1)
    index = (cursor - lag) % depth
    delayed = torch.gather(buffer, 1, index.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)

    noise = sensor["position_noise_m"].unsqueeze(1)
    measurement = delayed + torch.randn_like(delayed) * noise
    setattr(env, _SEAM_CACHE_ATTR, measurement)
    return measurement


def last_measured_target_error(
    env, command_name: str, asset_cfg: SceneEntityCfg, body_offset=TCP_OFFSET
) -> torch.Tensor:
    """Read the current seam measurement without advancing the history.

    The degraded observation term is the only writer: it pushes one sample per control
    step. A controller running before ``step()`` sees the sample produced at the end of
    the previous step, which is exactly what the policy was given, so the two consume
    the identical measurement. Pushing from here as well would advance the buffer twice
    per step and silently halve every delay.

    When no sample is cached the task does not degrade the seam channel at all, so the
    policy is reading the exact offset and the controller must read it too. Falling back
    to the degraded reading instead would hand the policy an exact measurement and the
    controller a late one, which looks like feedback control failing under delay when it
    is only the comparison being unfair.
    """
    cached = getattr(env, _SEAM_CACHE_ATTR, None)
    if cached is not None:
        return cached
    return target_error_w(env, command_name, asset_cfg, body_offset)


# --------------------------------------------------------------------------------------
# Free-floating vehicle (UVMS round).
#
# Everything above drives the base kinematically: `apply_immediate_base_motion` teleports
# the root every step, so the arm cannot influence it and the disturbance is exogenous.
# Below, the root is free and the only inputs to it are the hydrodynamic wrench applied
# here and the reaction PhysX produces from the arm's own motion. That reaction needs no
# code; it is what the round is about.
#
# The wave disturbance reuses `sample_base_motion`'s amplitude/frequency/phase draw, so
# a UVMS episode faces a disturbance drawn from the same distribution as a fixed-base
# one. It enters as a force rather than a pose - see `hydrodynamics.wave_wrench` for how
# an excursion in metres becomes newtons.
# --------------------------------------------------------------------------------------

_FREE_FLOATING_ATTR = "_marine_free_floating"
_DRAG_SCALE_ATTR = "_marine_drag_mismatch_scale"
_VEHICLE_ATTR = "_marine_vehicle_params"


def vehicle_params(env):
    """The nominal vehicle this environment carries."""
    from marine_manipulator import uvms_asset

    params = getattr(env, _VEHICLE_ATTR, None)
    if params is None:
        params = uvms_asset.default_vehicle_params()
        setattr(env, _VEHICLE_ATTR, params)
    return params


def drag_mismatch_scale(env) -> torch.Tensor:
    """Per-environment multiplier on the *simulation's* drag coefficients.

    The controllers and the policy always use the nominal coefficients. This factor is
    what makes their model wrong, and sweeping it is the Stage 3 x-axis. Defaults to
    ones, i.e. a perfectly known model, so a task that never configures the event runs
    at zero mismatch.
    """
    scale = getattr(env, _DRAG_SCALE_ATTR, None)
    if scale is None or scale.shape[0] != env.num_envs:
        scale = torch.ones(env.num_envs, device=env.device)
        setattr(env, _DRAG_SCALE_ATTR, scale)
    return scale


def sample_drag_mismatch(env, env_ids: torch.Tensor | None, scale_range: tuple[float, float]) -> None:
    """Draw a per-episode drag error. Log-uniform, so 0.5x and 2.0x are equally likely.

    Sampling uniformly in the coefficient would make 'drag twice as strong' four times
    more probable than 'drag half as strong', which biases a domain-randomised policy
    toward one side of an axis that is meant to be symmetric.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    scale = drag_mismatch_scale(env)
    low, high = (math.log(value) for value in scale_range)
    draw = torch.rand(int(env_ids.numel()), device=env.device) * (high - low) + low
    scale[env_ids] = torch.exp(draw)


def set_drag_mismatch(env, factor: float) -> None:
    """Pin the mismatch to one value, for the evaluation sweep."""
    drag_mismatch_scale(env)[:] = factor


def reset_uvms_root(
    env, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Return the free root to its default pose, moving with the wave it is released into.

    With a fixed base this was implicit. With a free one the hull carries its drift and
    its velocity across the episode boundary unless they are written back.

    The velocity written is not zero. The wave force is the second derivative of the
    same offset sinusoid the fixed-base round prescribed, and that trajectory has a
    non-zero velocity at ``t = 0``; a hull released at rest has to be accelerated onto
    it, and the missing velocity integrates into a ramp worth tens of centimetres over
    an episode. Starting the hull on the trajectory removes the artefact and makes the
    free-floating hull's excursion directly comparable with the prescribed one.
    """
    from marine_manipulator import hydrodynamics

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    default = asset.data.default_root_state[env_ids].clone()
    root_pose = default[:, :7].clone()
    root_pose[:, :3] = default[:, :3] + env.scene.env_origins[env_ids]

    motion = _state(env)
    velocity = hydrodynamics.wave_velocity(
        motion["amplitude"][env_ids], motion["frequency"][env_ids], motion["phase"][env_ids]
    )
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocity, env_ids=env_ids)
    asset.update(0.0)


def _default_root_pose(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_root_state
    return default[:, :3] + env.scene.env_origins, default[:, 3:7]


def uvms_base_pose(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Where the hull has drifted to, ``(num_envs, 6)``: translation then roll/pitch/yaw.

    Expressed as a displacement from the pose the episode started at, so it is directly
    comparable with the fixed-base task's `immediate_base_pose`, which was also an
    offset from a neutral pose. That keeps the observation scaling meaningful and makes
    the two rounds' hull excursions readable against each other.
    """
    from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate

    asset = env.scene[asset_cfg.name]
    origin_pos, origin_quat = _default_root_pose(env, asset_cfg)
    delta_quat = quat_mul(quat_conjugate(origin_quat), asset.data.root_quat_w)
    roll, pitch, yaw = euler_xyz_from_quat(delta_quat)
    angles = torch.stack((roll, pitch, yaw), dim=1)
    # `euler_xyz_from_quat` returns [0, 2pi); wrap so a small negative roll reads as a
    # small negative number rather than as ~6.28, which would saturate the observation.
    angles = torch.atan2(torch.sin(angles), torch.cos(angles))
    return torch.cat((asset.data.root_pos_w - origin_pos, angles), dim=1)


def uvms_base_velocity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Hull linear and angular velocity in the world frame, ``(num_envs, 6)``."""
    asset = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_lin_vel_w, asset.data.root_ang_vel_w), dim=1)


def normalized_uvms_base_pose(env) -> torch.Tensor:
    _, amp_high, _, _ = _range_tensors(env.device)
    return torch.clamp(uvms_base_pose(env) / amp_high, -1.0, 1.0)


def normalized_uvms_base_velocity(env) -> torch.Tensor:
    _, amp_high, _, freq_high = _range_tensors(env.device)
    scale = amp_high * 2.0 * math.pi * freq_high
    return torch.clamp(uvms_base_velocity(env) / scale, -1.0, 1.0)


def root_pose_for_controller(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The base pose a model-based controller reads, ``(pos, quat)``.

    On a free-floating task the hull's pose is a state to be measured, not a signal to
    be reconstructed from a known sinusoid, so the analytic reconstruction in
    :func:`measured_root_pose` does not apply and would return the *undisturbed* pose.
    This round degrades no sensors - that was the previous round's axis - so the
    controller reads the true pose, which is also exactly what the policy observes.
    """
    if getattr(env, _FREE_FLOATING_ATTR, False):
        asset = env.scene[asset_cfg.name]
        return asset.data.root_pos_w, asset.data.root_quat_w
    return measured_root_pose(env, asset_cfg)


def hydrodynamic_wrench(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> tuple[torch.Tensor, torch.Tensor]:
    """Total body-frame force and torque on the hull: drag, restoring and waves.

    The drag coefficients used here are the nominal ones scaled by this environment's
    mismatch factor, so the *simulation* runs a vehicle whose drag differs from the one
    the controller and the policy believe in.
    """
    from marine_manipulator import hydrodynamics

    asset = env.scene[asset_cfg.name]
    params = vehicle_params(env)
    device = env.device

    scale = drag_mismatch_scale(env).unsqueeze(1)
    linear = torch.tensor(params.linear_drag, device=device).unsqueeze(0) * scale
    quadratic = torch.tensor(params.quadratic_drag, device=device).unsqueeze(0) * scale
    drag_force, drag_torque = hydrodynamics.drag_wrench(
        asset.data.root_lin_vel_b, asset.data.root_ang_vel_b, linear, quadratic
    )

    restore_force, restore_torque = hydrodynamics.restoring_wrench(
        asset.data.root_quat_w,
        params.mass_kg,
        params.cob_offset_b,
        params.net_buoyancy_fraction,
    )

    motion = _state(env)
    elapsed = env.episode_length_buf.to(torch.float32).unsqueeze(1) * env.step_dt
    wave_force_w, wave_torque_w = hydrodynamics.wave_wrench(
        elapsed,
        motion["amplitude"],
        motion["frequency"],
        motion["phase"],
        params.effective_spatial_inertia,
    )
    # Waves act in world axes - a swell does not roll with the vehicle - so they are
    # rotated into the body frame the wrench composer expects.
    wave_force = quat_apply_inverse(asset.data.root_quat_w, wave_force_w)
    wave_torque = quat_apply_inverse(asset.data.root_quat_w, wave_torque_w)

    return (
        drag_force + restore_force + wave_force,
        drag_torque + restore_torque + wave_torque,
    )


def apply_hydrodynamic_wrench(
    env, env_ids: torch.Tensor | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    """Write the hull wrench into the articulation's permanent wrench buffer.

    Set once per *control* step, so it is held constant across the decimation's physics
    substeps. At 30 Hz against a drag time constant of order a second that zero-order
    hold is immaterial, but it is an approximation and is recorded as one.

    The wrench composer is used directly rather than through
    ``set_external_force_and_torque``, which logs a deprecation warning on every call
    and would emit one thirty times a second for the length of a training run.
    """
    asset = env.scene[asset_cfg.name]
    base_index = getattr(env, "_marine_base_body_index", None)
    if base_index is None:
        base_index = asset.body_names.index("base_link")
        setattr(env, "_marine_base_body_index", base_index)

    force, torque = hydrodynamic_wrench(env, asset_cfg)
    body_ids = torch.tensor([base_index], dtype=torch.int32, device=env.device)
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=force.unsqueeze(1).contiguous(),
        torques=torque.unsqueeze(1).contiguous(),
        body_ids=body_ids,
        is_global=False,
    )
