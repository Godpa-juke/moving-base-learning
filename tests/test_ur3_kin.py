"""Validate the analytic UR3 kinematics against the simulator-derived calibration poses."""

from __future__ import annotations

import math

import torch

from marine_manipulator import calibration as mdp
from marine_manipulator import ur3_kin

TOOL = mdp.TCP_OFFSET[2]


def _q(*rows: tuple[float, ...]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float64)


def test_forward_kinematics_matches_simulated_tcp() -> None:
    """Both configurations solved inside Isaac must reproduce from the DH table."""
    q = _q(mdp.PRECISION_START_JOINT_POS, mdp.PRECISION_FAR_START_JOINT_POS)
    expected = torch.tensor(
        [mdp.PRECISION_START_TCP_E, mdp.PRECISION_FAR_START_TCP_E], dtype=torch.float64
    )
    tcp = ur3_kin.tcp_position(q, TOOL)
    error = torch.linalg.norm(tcp - expected, dim=1)
    # The start pose is exact; the far pose carries the residual of the numerical
    # solver that originally produced it inside the simulator.
    assert float(error[0]) < 1e-5, error
    assert float(error[1]) < 2e-3, error


def test_jacobian_matches_finite_differences() -> None:
    torch.manual_seed(0)
    q = (torch.rand((8, 6), dtype=torch.float64) - 0.5) * 3.0
    analytic = ur3_kin.jacobian(q, TOOL)
    eps = 1e-6
    for joint in range(6):
        delta = torch.zeros_like(q)
        delta[:, joint] = eps
        forward = ur3_kin.tcp_position(q + delta, TOOL)
        backward = ur3_kin.tcp_position(q - delta, TOOL)
        numeric = (forward - backward) / (2 * eps)
        assert torch.allclose(analytic[:, :3, joint], numeric, atol=1e-6), joint


def test_inverse_kinematics_round_trip() -> None:
    torch.manual_seed(1)
    nominal = _q(mdp.PRECISION_START_JOINT_POS).repeat(64, 1)
    perturbed = nominal + (torch.rand_like(nominal) - 0.5) * 0.8
    flange = ur3_kin.forward_kinematics(perturbed)
    target_position = flange[:, :3, 3] + TOOL * flange[:, :3, 2]
    q, converged = ur3_kin.inverse_kinematics(
        target_position, flange[:, :3, :3], nominal, TOOL
    )
    assert bool(converged.all()), int((~converged).sum())
    residual = torch.linalg.norm(ur3_kin.tcp_position(q, TOOL) - target_position, dim=1)
    assert float(residual.max()) < 1e-4, float(residual.max())


def test_inverse_kinematics_reaches_vertical_tool_targets() -> None:
    """The task pose: tip on the line, cylinder axis pointing straight down."""
    count = 512
    torch.manual_seed(2)
    target = torch.empty((count, 3), dtype=torch.float64)
    target[:, 0] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_X)
    target[:, 1] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_Y)
    target[:, 2] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_Z)
    axis = torch.zeros((count, 3), dtype=torch.float64)
    axis[:, 2] = -1.0
    rotation = ur3_kin.rotation_from_tool_axis(axis)
    seed = _q(mdp.VERTICAL_TOOL_IK_SEED).repeat(count, 1)
    q, converged = ur3_kin.inverse_kinematics(
        target, rotation, seed, TOOL, iterations=300,
        position_tolerance=1e-5, rotation_tolerance=1e-4,
    )
    assert bool(converged.all()), int((~converged).sum())
    residual = torch.linalg.norm(ur3_kin.tcp_position(q, TOOL) - target, dim=1)
    assert float(residual.max()) < 1e-5, float(residual.max())
    tilt = ur3_kin.tool_axis_error(q, axis)
    assert float(tilt.max()) < math.radians(0.01), math.degrees(float(tilt.max()))


def test_dedicated_seed_beats_nominal_postures() -> None:
    """Guards the reason VERTICAL_TOOL_IK_SEED exists rather than a natural posture.

    Nominal postures do reach some of these targets, but the hit rate swings with
    the sampled batch (5-80% observed), so seeding from one is not dependable.
    """
    count = 256
    torch.manual_seed(3)
    target = torch.empty((count, 3), dtype=torch.float64)
    target[:, 0] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_X)
    target[:, 1] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_Y)
    target[:, 2] = torch.empty(count, dtype=torch.float64).uniform_(*mdp.TARGET_BOX_Z)
    axis = torch.zeros((count, 3), dtype=torch.float64)
    axis[:, 2] = -1.0
    rotation = ur3_kin.rotation_from_tool_axis(axis)

    def hit_rate(seed_q) -> float:
        _, converged = ur3_kin.inverse_kinematics(
            target, rotation, _q(seed_q).repeat(count, 1), TOOL, iterations=300
        )
        return float(converged.double().mean())

    dedicated = hit_rate(mdp.VERTICAL_TOOL_IK_SEED)
    assert dedicated == 1.0, dedicated
    for nominal in (mdp.PRECISION_START_JOINT_POS, mdp.PRECISION_FAR_START_JOINT_POS):
        assert hit_rate(nominal) < dedicated, nominal


def test_wrap_to_reference_preserves_pose() -> None:
    torch.manual_seed(4)
    q = (torch.rand((32, 6), dtype=torch.float64) - 0.5) * 4.0 * math.pi
    reference = torch.zeros_like(q)
    wrapped = ur3_kin.wrap_to_reference(q, reference)
    assert float(wrapped.abs().max()) <= math.pi + 1e-9, float(wrapped.abs().max())
    before = ur3_kin.tcp_position(q, TOOL)
    after = ur3_kin.tcp_position(wrapped, TOOL)
    assert torch.allclose(before, after, atol=1e-9)
