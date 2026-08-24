"""Batched UR3 kinematics: forward kinematics, geometric Jacobian, damped-least-squares IK.

The DH table below is the published Universal Robots UR3 table. It was validated
against the Isaac Lab ``ur3.usd`` articulation using two independently solved
configurations recorded in :mod:`marine_manipulator.tasks.random_base_line.mdp`;
both reproduce the simulated cylinder-tip TCP to 1e-5 m once the constant base
rotation ``BASE_YAW_OFFSET`` is applied.

Frames
------
``T_dh``    base -> flange, built from the DH table.
``T_robot`` base -> flange in the articulation's root frame, i.e. ``Rz(pi) @ T_dh``.

The tool is a cylinder along the flange ``+Z`` axis, so it is rotationally
symmetric: only the *direction* of that axis is physically meaningful, never the
roll about it. IK still solves a full 6-DoF pose (roll picked arbitrarily but
consistently) because that is numerically better conditioned; rewards and metrics
should use :func:`tool_axis_error` rather than a quaternion difference.
"""

from __future__ import annotations

import math

import torch

# Universal Robots UR3 DH parameters (metres).
D1 = 0.1519
A2 = -0.24365
A3 = -0.21325
D4 = 0.11235
D5 = 0.08535
D6 = 0.0819

#: The USD articulation root is rotated by pi about Z relative to the DH base frame.
BASE_YAW_OFFSET = math.pi

# (d, a, alpha) per joint; theta comes from the joint state.
_DH = (
    (D1, 0.0, math.pi / 2),
    (0.0, A2, 0.0),
    (0.0, A3, 0.0),
    (D4, 0.0, math.pi / 2),
    (D5, 0.0, -math.pi / 2),
    (D6, 0.0, 0.0),
)

JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

JOINT_LIMIT = 2.0 * math.pi


def _link_transforms(q: torch.Tensor) -> torch.Tensor:
    """Cumulative base->joint-i transforms, shape ``(batch, 7, 4, 4)``.

    Index 0 is the identity (base frame) and index 6 is the flange, both expressed
    in the robot root frame.
    """
    batch = q.shape[0]
    transforms = torch.zeros((batch, 7, 4, 4), dtype=q.dtype, device=q.device)
    accumulated = torch.eye(4, dtype=q.dtype, device=q.device).expand(batch, 4, 4).clone()
    # Rotate the DH base into the articulation root frame once, up front.
    yaw = torch.full((batch,), BASE_YAW_OFFSET, dtype=q.dtype, device=q.device)
    accumulated[:, 0, 0] = torch.cos(yaw)
    accumulated[:, 0, 1] = -torch.sin(yaw)
    accumulated[:, 1, 0] = torch.sin(yaw)
    accumulated[:, 1, 1] = torch.cos(yaw)
    transforms[:, 0] = accumulated
    for index, (d, a, alpha) in enumerate(_DH):
        theta = q[:, index]
        ct, st = torch.cos(theta), torch.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        step = torch.zeros((batch, 4, 4), dtype=q.dtype, device=q.device)
        step[:, 0, 0] = ct
        step[:, 0, 1] = -st * ca
        step[:, 0, 2] = st * sa
        step[:, 0, 3] = a * ct
        step[:, 1, 0] = st
        step[:, 1, 1] = ct * ca
        step[:, 1, 2] = -ct * sa
        step[:, 1, 3] = a * st
        step[:, 2, 1] = sa
        step[:, 2, 2] = ca
        step[:, 2, 3] = d
        step[:, 3, 3] = 1.0
        accumulated = accumulated @ step
        transforms[:, index + 1] = accumulated
    return transforms


def forward_kinematics(q: torch.Tensor) -> torch.Tensor:
    """Base->flange homogeneous transform in the robot root frame, ``(batch, 4, 4)``."""
    return _link_transforms(q)[:, 6]


def tcp_position(q: torch.Tensor, tool_offset: float) -> torch.Tensor:
    """Cylinder-tip position in the robot root frame, ``(batch, 3)``."""
    flange = forward_kinematics(q)
    return flange[:, :3, 3] + tool_offset * flange[:, :3, 2]


def tool_axis(q: torch.Tensor) -> torch.Tensor:
    """Unit cylinder axis (flange +Z) in the robot root frame, ``(batch, 3)``."""
    return forward_kinematics(q)[:, :3, 2]


def jacobian(q: torch.Tensor, tool_offset: float) -> torch.Tensor:
    """Geometric Jacobian at the cylinder tip, ``(batch, 6, 6)``.

    Rows 0-2 map joint rates to tip linear velocity, rows 3-5 to angular velocity.
    """
    transforms = _link_transforms(q)
    tip = transforms[:, 6, :3, 3] + tool_offset * transforms[:, 6, :3, 2]
    axes = transforms[:, :6, :3, 2]  # joint i rotates about z of frame i-1
    origins = transforms[:, :6, :3, 3]
    linear = torch.cross(axes, tip.unsqueeze(1) - origins, dim=2)
    return torch.cat((linear, axes), dim=2).transpose(1, 2)


def _rotation_error(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Axis-angle vector rotating ``current`` onto ``target``, ``(batch, 3)``."""
    relative = target @ current.transpose(1, 2)
    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cos_angle)
    axis = torch.stack(
        (
            relative[:, 2, 1] - relative[:, 1, 2],
            relative[:, 0, 2] - relative[:, 2, 0],
            relative[:, 1, 0] - relative[:, 0, 1],
        ),
        dim=1,
    )
    norm = torch.linalg.norm(axis, dim=1, keepdim=True)
    # Near 0 and pi the cross-product form degenerates; fall back to the raw axis,
    # which still points the right way for the small steps DLS takes.
    safe = torch.where(norm > 1e-8, axis / norm.clamp(min=1e-8), axis)
    return safe * angle.unsqueeze(1)


def tool_axis_error(q: torch.Tensor, desired_axis: torch.Tensor) -> torch.Tensor:
    """Angle in radians between the cylinder axis and ``desired_axis``, ``(batch,)``."""
    axis = tool_axis(q)
    desired = desired_axis / torch.linalg.norm(desired_axis, dim=1, keepdim=True)
    return torch.acos((axis * desired).sum(dim=1).clamp(-1.0, 1.0))


def inverse_kinematics(
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    q_seed: torch.Tensor,
    tool_offset: float,
    iterations: int = 80,
    damping: float = 0.05,
    step_clamp: float = 0.2,
    position_tolerance: float = 1e-4,
    rotation_tolerance: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Damped-least-squares IK for the cylinder tip.

    Args:
        target_position: desired tip position in the robot root frame, ``(batch, 3)``.
        target_rotation: desired flange rotation matrix, ``(batch, 3, 3)``.
        q_seed: initial joint configuration, ``(batch, 6)``. Determines which
            elbow/shoulder branch the solver lands on, so pass the nominal posture.
        tool_offset: cylinder length along the flange +Z axis.

    Returns:
        ``(q, converged)`` where ``converged`` is a bool mask over the batch.
    """
    q = q_seed.clone()
    identity = torch.eye(6, dtype=q.dtype, device=q.device).expand(q.shape[0], 6, 6)
    converged = torch.zeros(q.shape[0], dtype=torch.bool, device=q.device)
    for _ in range(iterations):
        transforms = _link_transforms(q)
        flange = transforms[:, 6]
        tip = flange[:, :3, 3] + tool_offset * flange[:, :3, 2]
        position_error = target_position - tip
        rotation_error = _rotation_error(flange[:, :3, :3], target_rotation)
        converged = (torch.linalg.norm(position_error, dim=1) < position_tolerance) & (
            torch.linalg.norm(rotation_error, dim=1) < rotation_tolerance
        )
        if bool(converged.all()):
            break
        error = torch.cat((position_error, rotation_error), dim=1)
        jac = jacobian(q, tool_offset)
        jjt = jac @ jac.transpose(1, 2) + (damping**2) * identity
        delta = jac.transpose(1, 2) @ torch.linalg.solve(jjt, error.unsqueeze(2))
        delta = delta.squeeze(2).clamp(-step_clamp, step_clamp)
        # Frozen once converged so solved environments stop drifting.
        q = torch.where(converged.unsqueeze(1), q, q + delta)
        q = q.clamp(-JOINT_LIMIT, JOINT_LIMIT)
    return q, converged


def rotation_from_tool_axis(axis: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
    """Build a flange rotation whose +Z is ``axis``, ``(batch, 3, 3)``.

    Roll about the tool axis is unobservable for a cylinder, so it is fixed by
    projecting ``reference`` (default world +X) onto the plane normal to ``axis``.
    """
    z = axis / torch.linalg.norm(axis, dim=1, keepdim=True)
    if reference is None:
        reference = torch.zeros_like(z)
        reference[:, 0] = 1.0
    projected = reference - (reference * z).sum(dim=1, keepdim=True) * z
    degenerate = torch.linalg.norm(projected, dim=1) < 1e-6
    if bool(degenerate.any()):
        fallback = torch.zeros_like(z)
        fallback[:, 1] = 1.0
        fallback = fallback - (fallback * z).sum(dim=1, keepdim=True) * z
        projected = torch.where(degenerate.unsqueeze(1), fallback, projected)
    x = projected / torch.linalg.norm(projected, dim=1, keepdim=True)
    y = torch.cross(z, x, dim=1)
    return torch.stack((x, y, z), dim=2)


def solve_with_restarts(
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    seeds: torch.Tensor,
    tool_offset: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run :func:`inverse_kinematics` from several seeds and keep the first success.

    Seeding matters a lot here: the vertical-tool solutions sit in a joint-space
    branch far from any nominal posture, and damped least squares stalls at a
    non-solution when it starts in the wrong basin. Hit rates from a nominal
    posture swing between roughly 5% and 80% depending on the batch of targets.

    Args:
        seeds: candidate configurations, ``(num_seeds, 6)``, tried in order.

    Returns:
        ``(q, converged)`` with the same shapes as :func:`inverse_kinematics`.
    """
    batch = target_position.shape[0]
    best = seeds[0].to(target_position).expand(batch, 6).clone()
    solved = torch.zeros(batch, dtype=torch.bool, device=target_position.device)
    for seed in seeds:
        if bool(solved.all()):
            break
        start = seed.to(target_position).expand(batch, 6).clone()
        q, converged = inverse_kinematics(
            target_position, target_rotation, start, tool_offset, **kwargs
        )
        take = converged & ~solved
        best = torch.where(take.unsqueeze(1), q, best)
        solved |= converged
    return best, solved


def sample_seed_bank(
    target_position: torch.Tensor,
    target_rotation: torch.Tensor,
    tool_offset: float,
    candidates: int = 4096,
    keep: int = 8,
    generator: torch.Generator | None = None,
    **kwargs,
) -> torch.Tensor:
    """Search joint space for a small set of seeds that covers a target distribution.

    Intended for offline use: run it once over a representative sample of targets,
    hard-code the result, and pass it to :func:`solve_with_restarts` at reset time
    so the online cost stays at a handful of damped-least-squares iterations.
    """
    device = target_position.device
    pool = (
        torch.rand((candidates, 6), dtype=target_position.dtype, device=device, generator=generator)
        * 2.0
        - 1.0
    ) * math.pi
    chosen: list[torch.Tensor] = []
    remaining = torch.ones(target_position.shape[0], dtype=torch.bool, device=device)
    for _ in range(keep):
        if not bool(remaining.any()):
            break
        best_seed = None
        best_score = -1
        for seed in pool:
            start = seed.expand(int(remaining.sum()), 6).clone()
            _, converged = inverse_kinematics(
                target_position[remaining], target_rotation[remaining], start, tool_offset, **kwargs
            )
            score = int(converged.sum())
            if score > best_score:
                best_score, best_seed = score, seed
        if best_seed is None or best_score == 0:
            break
        chosen.append(best_seed)
        start = best_seed.expand(target_position.shape[0], 6).clone()
        _, converged = inverse_kinematics(
            target_position, target_rotation, start, tool_offset, **kwargs
        )
        remaining &= ~converged
    return torch.stack(chosen) if chosen else pool[:1]


def wrap_to_reference(q: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Shift each joint by whole turns to land within pi of ``reference``.

    A 2*pi turn leaves the pose untouched, but IK routinely returns wound-up
    branches (an elbow at 5.9 rad rather than -0.38 rad). Those are equivalent
    kinematically yet sit far from the articulation's default pose, which is the
    origin of the joint-position action space.
    """
    return q - 2.0 * math.pi * torch.round((q - reference) / (2.0 * math.pi))
