"""Fossen's rigid-body-in-fluid model, applied as an external wrench on the hull.

Three terms, handled three different ways, for reasons that are worth stating because
each one is a modelling decision the results depend on.

**Added mass** is folded into the rigid body's mass and inertia rather than applied as
a force. Applying ``-M_A v̇`` explicitly needs the acceleration, which is only available
by finite difference, and differentiating a simulated velocity at 60 Hz and feeding it
back as a force is a positive feedback loop that diverges. Diagonal absorption is the
standard practical treatment. It costs two approximations: the off-diagonal coupling
terms are dropped, and — because PhysX carries a *scalar* rigid-body mass — the
translational added mass has to be isotropic, so the surge/sway/heave anisotropy of a
real hull is replaced by its mean. Rotational added inertia stays anisotropic, since
USD does carry a diagonal inertia tensor.

**Drag** is an explicit wrench, evaluated from the hull's own velocity. Stable, because
it is dissipative and depends on velocity rather than acceleration.

**Restoring** — weight against buoyancy — is also explicit, and this is a deliberate
departure from ``docs/UVMS_PLAN.md``, which asked for PhysX gravity to be switched on
instead. Gravity is left *off* on the articulation and the whole restoring vector is
applied here. Two reasons:

1. Folding added mass into the rigid-body mass inflates it by ~70%. PhysX gravity then
   pulls on the inflated mass while buoyancy can only balance the true displaced
   weight, leaving a spurious 0.7·m·g sink rate. Applying weight explicitly against the
   *true* mass removes the contradiction at its source.
2. ``disable_gravity`` is a property of the spawned USD, so it cannot be set per link.
   With PhysX gravity on, the arm links would weigh their full dry weight, which a
   subsea arm designed to be near neutrally buoyant does not. Leaving gravity off
   models the arm as neutrally buoyant — the same assumption the fixed-base round ran
   under, which keeps the two rounds comparable.

The restoring moment that the plan wanted from the centre-of-buoyancy offset is
reproduced exactly: buoyancy acts at the centre of buoyancy and weight at the centre of
gravity, and the offset between them produces the roll and pitch righting moments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

#: Standard gravity, m/s^2.
GRAVITY = 9.81

AXIS_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


@dataclass(frozen=True)
class VehicleParams:
    """Hydrodynamic and inertial description of the lumped hull.

    Drag is stored as the six diagonal entries of Fossen's ``D_l`` and ``D_q``, in the
    body frame, ordered surge, sway, heave, roll, pitch, yaw. Linear entries are
    N·s/m and N·m·s/rad; quadratic entries are N/(m/s)^2 and N·m/(rad/s)^2.
    """

    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]

    #: Translational added mass as a fraction of the dry mass. Scalar, because PhysX
    #: carries one mass per body; see the module docstring.
    added_mass_fraction: float

    #: Rotational added inertia as a fraction of the dry inertia, per body axis.
    added_inertia_fraction: tuple[float, float, float]

    linear_drag: tuple[float, float, float, float, float, float]
    quadratic_drag: tuple[float, float, float, float, float, float]

    #: Centre of buoyancy relative to the centre of gravity, body frame, metres.
    #: A positive z is the usual stable arrangement: buoyancy above weight.
    cob_offset_b: tuple[float, float, float]

    #: Buoyancy in excess of weight, as a fraction of weight. Nominally zero: with no
    #: thrusters (plan stage 1c) nothing opposes a net buoyancy, so any non-zero value
    #: accumulates into an unbounded drift over an episode.
    net_buoyancy_fraction: float = 0.0

    def scaled_drag(self, factor: float) -> "VehicleParams":
        """The same vehicle with every drag coefficient multiplied by ``factor``.

        This is the Stage 3 mismatch axis. It scales the linear and the quadratic
        coefficients together, so a single number describes 'the drag model is wrong by
        this much' rather than two independently wrong models.
        """
        return replace(
            self,
            linear_drag=tuple(value * factor for value in self.linear_drag),
            quadratic_drag=tuple(value * factor for value in self.quadratic_drag),
        )

    @property
    def effective_mass_kg(self) -> float:
        """Dry mass plus isotropic translational added mass."""
        return self.mass_kg * (1.0 + self.added_mass_fraction)

    @property
    def effective_inertia_kg_m2(self) -> tuple[float, float, float]:
        """Dry inertia plus per-axis rotational added inertia."""
        return tuple(
            value * (1.0 + fraction)
            for value, fraction in zip(self.inertia_kg_m2, self.added_inertia_fraction)
        )

    @property
    def effective_spatial_inertia(self) -> tuple[float, ...]:
        """The six diagonal entries a wrench has to accelerate, in wrench order.

        Used to convert a desired hull *excursion* into the force that produces it, so
        that the wave disturbance can be specified in metres — the same units the
        fixed-base round's prescribed motion used — rather than in newtons.
        """
        return (self.effective_mass_kg,) * 3 + self.effective_inertia_kg_m2


def _as_batch(values, num_envs: int, device) -> torch.Tensor:
    """Broadcast a constant tuple, or pass a per-environment tensor through."""
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.float32)
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0).expand(num_envs, -1)


def drag_wrench(
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    linear_drag: torch.Tensor | tuple,
    quadratic_drag: torch.Tensor | tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fossen's ``-(D_l v + D_q |v| v)``, evaluated in the body frame.

    ``|v| v`` rather than ``v^2`` so the quadratic term stays dissipative when the
    velocity reverses; squaring alone would make drag push the hull forwards in reverse.
    """
    velocity = torch.cat((lin_vel_b, ang_vel_b), dim=1)
    num_envs, device = velocity.shape[0], velocity.device
    d_l = _as_batch(linear_drag, num_envs, device)
    d_q = _as_batch(quadratic_drag, num_envs, device)
    wrench = -(d_l * velocity + d_q * velocity.abs() * velocity)
    return wrench[:, :3], wrench[:, 3:]


def restoring_wrench(
    root_quat_w: torch.Tensor,
    mass_kg: float,
    cob_offset_b: torch.Tensor | tuple,
    net_buoyancy_fraction: float = 0.0,
    gravity: float = GRAVITY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weight at the centre of gravity against buoyancy at the centre of buoyancy.

    Returned in the body frame, because that is the frame the wrench composer applies
    an ``is_global=False`` wrench in, and because the moment arm is a body-frame
    quantity. The centre of gravity is taken as the body frame's origin — the vehicle
    is lumped into ``base_link``, so its centre of gravity *is* that link's origin.
    """
    from isaaclab.utils.math import quat_apply_inverse

    num_envs, device = root_quat_w.shape[0], root_quat_w.device
    weight = mass_kg * gravity
    buoyancy = weight * (1.0 + net_buoyancy_fraction)

    up_w = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    up_w[:, 2] = 1.0
    up_b = quat_apply_inverse(root_quat_w, up_w)

    buoyancy_force_b = up_b * buoyancy
    force_b = up_b * (buoyancy - weight)
    arm = _as_batch(cob_offset_b, num_envs, device)
    # Weight acts at the origin and so contributes no moment; only buoyancy does.
    torque_b = torch.cross(arm, buoyancy_force_b, dim=1)
    return force_b, torque_b


def wave_wrench(
    elapsed_s: torch.Tensor,
    amplitude: torch.Tensor,
    frequency_hz: torch.Tensor,
    phase: torch.Tensor,
    spatial_inertia: torch.Tensor | tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sinusoidal disturbance, specified as an excursion and converted to a force.

    The fixed-base round prescribed the hull's pose directly, as
    ``x(t) = 0.5 A (sin(ωt + φ) − sin(φ))`` — an offset sinusoid that starts at the
    neutral pose. Here the hull is free, so the same disturbance has to enter as a
    force, and the force that produces exactly that trajectory is ``m ẍ``, which is
    what this returns. Specifying the disturbance as an excursion rather than in
    newtons is what makes the two rounds' disturbances comparable.

    Producing that trajectory also requires the initial velocity ``ẋ(0)``, which is not
    zero: see :func:`wave_velocity`. A hull released at rest under this force instead
    integrates the missing velocity into a slow ramp, which shows up as tens of
    centimetres of drift per episode and is a numerical artefact rather than a sea
    state.

    Drag makes the realised excursion smaller than the requested one, which is correct:
    a hull in water is harder to shake than the same hull in vacuum.

    All inputs are ``(num_envs, 6)`` except ``elapsed_s``, which is ``(num_envs, 1)``.
    """
    num_envs, device = amplitude.shape[0], amplitude.device
    inertia = _as_batch(spatial_inertia, num_envs, device)
    omega = 2.0 * torch.pi * frequency_hz
    wrench = -0.5 * inertia * amplitude * omega.square() * torch.sin(omega * elapsed_s + phase)
    return wrench[:, :3], wrench[:, 3:]


def wave_velocity(
    amplitude: torch.Tensor, frequency_hz: torch.Tensor, phase: torch.Tensor
) -> torch.Tensor:
    """The hull velocity at ``t = 0`` consistent with :func:`wave_wrench`.

    ``d/dt [0.5 A (sin(ωt + φ) − sin(φ))]`` at ``t = 0``. Writing this into the root at
    reset puts the hull onto the disturbance's trajectory instead of leaving it to be
    accelerated onto it, which removes the ramp described above. The rotational entries
    are Euler rates used as body angular rates, exact only at small angles — the same
    approximation the fixed-base round's prescribed motion made.
    """
    return 0.5 * amplitude * (2.0 * torch.pi * frequency_hz) * torch.cos(phase)
