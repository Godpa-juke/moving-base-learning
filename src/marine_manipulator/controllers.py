"""The two analytic controllers the learned policy is measured against.

Baseline A (:class:`AnalyticIkController`) is the previous round's resolved-rate
controller, unchanged. It reads the tool-to-line offset and inverts the *arm's*
Jacobian, which silently treats the base it is bolted to as an anchor.

Baseline B (:class:`CoupledUvmsController`) is this round's real opponent. On a free
vehicle the anchor assumption is wrong twice over — the arm's own command pushes the
hull back, and the hull drifts under the water's forces regardless — and B models both.

They live here rather than inside ``scripts/evaluate.py`` because two scripts run them:
the single-condition evaluation and the mismatch sweep. A baseline that exists in two
copies is a baseline that will eventually be two different baselines, and the plan's
rule 2 — the baseline gets exactly the information the policy gets, no more and no
less — is not checkable if there is more than one of it.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from marine_manipulator.tasks.random_base_line import mdp


class AnalyticIkController:
    """Model-based baseline: solve IK against the true base pose every step.

    This is the controller a reviewer will ask about. It is given strictly more
    information than the policy (the exact root pose rather than a normalised
    observation of the sampled motion) and drives the same action term, so it
    passes through the identical rate limiter, action clipping and joint PD. Any
    error it leaves is therefore attributable to actuation limits and the
    one-step delay inherent in reacting to a disturbance, not to modelling error.
    """

    def __init__(self, env, iterations: int = 20, mode: str = "seam", gain: float = 1.0):
        from isaaclab.managers import SceneEntityCfg
        from marine_manipulator import calibration, ur3_kin

        self._env = env
        self._kin = ur3_kin
        self._iterations = iterations
        self._mode = mode
        self._gain = gain
        self._tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self._tool_cfg.resolve(env.scene)
        self._robot = env.scene["robot"]
        self._command = env.command_manager._terms["ee_pose"]
        self._term = env.action_manager._terms["arm_action"]
        self._tool = calibration.TCP_OFFSET[2]
        self._scale = calibration.REALISTIC_ACTION_SCALE_RAD
        self._down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(
            env.num_envs, 3
        )
        self.solved: list[float] = []

    def __call__(self, obs):
        if self._mode == "seam":
            return self._seam_step()
        return self._pose_step()

    def _seam_step(self):
        """Resolved-rate step driven by the measured tool-to-line offset.

        Only the base *orientation* enters, through the frame the Jacobian is
        expressed in; a late base *position* cannot mislead this controller the way
        it misleads the open-loop form. That is the point of using it: whatever
        advantage remains for the policy then has to come from prediction.
        """
        error_w = mdp.last_measured_target_error(
            self._env, "ee_pose", self._tool_cfg, mdp.TCP_OFFSET
        )
        _, root_quat = mdp.root_pose_for_controller(self._env)
        error_root = quat_apply_inverse(root_quat, error_w)

        joints = self._robot.data.joint_pos.clone()
        desired_axis = quat_apply_inverse(root_quat, self._down)
        current_axis = self._kin.tool_axis(joints)
        # Axis-angle taking the cylinder axis onto its target. The cross product
        # carries no roll component, which is correct: the tool is axially symmetric.
        axis = torch.cross(current_axis, desired_axis, dim=1)
        sine = torch.linalg.norm(axis, dim=1, keepdim=True)
        cosine = (current_axis * desired_axis).sum(dim=1, keepdim=True)
        angle = torch.atan2(sine, cosine)
        rotation_error = axis / sine.clamp(min=1e-8) * angle

        error = torch.cat((error_root, rotation_error), dim=1) * self._gain
        # Residual policy learning perturbs the *input* to the controller rather than
        # its output (Silver et al. 2018; Johannink et al. 2019), which is how Herland &
        # Bach apply it. Correcting in task space leaves the kinematics to the controller
        # and keeps the correction interpretable in metres. Absent on any task that does
        # not set it, so plain baseline runs are unaffected.
        residual = getattr(self._env, "_marine_task_residual", None)
        if residual is not None:
            error = error + residual
        jac = self._kin.jacobian(joints, self._tool)
        identity = torch.eye(6, device=jac.device, dtype=jac.dtype).expand_as(jac)
        damped = jac @ jac.transpose(1, 2) + (0.05**2) * identity
        delta = jac.transpose(1, 2) @ torch.linalg.solve(damped, error.unsqueeze(2))
        command = joints + delta.squeeze(2)
        return (command - self._term._offset) / self._scale

    def _pose_step(self):
        target_w = self._env.scene.env_origins + self._command.pose_command_b[:, :3]
        # The same degraded measurement the policy observes. On a task without
        # sensor degradation this returns the true root pose exactly, so the
        # clean-condition numbers are unaffected.
        root_pos, root_quat = mdp.root_pose_for_controller(self._env)
        target_root = quat_apply_inverse(root_quat, target_w - root_pos)
        axis_root = quat_apply_inverse(root_quat, self._down)
        # Warm-started from the measured configuration, so a couple of
        # damped-least-squares steps suffice and the branch never jumps.
        seed = self._robot.data.joint_pos.clone()
        joints, converged = self._kin.inverse_kinematics(
            target_root,
            self._kin.rotation_from_tool_axis(axis_root),
            seed,
            self._tool,
            iterations=self._iterations,
        )
        joints = self._kin.wrap_to_reference(joints, seed)
        self.solved.append(float(converged.double().mean()))
        return (joints - self._term._offset) / self._scale

class CoupledUvmsController:
    """Baseline B: resolved-rate control that knows the base is not standing still.

    Baseline A (``--controller ik``) reads the tool-to-line offset and inverts the
    *arm's* Jacobian, which silently assumes the hull is a fixed anchor. On a free
    vehicle that assumption is wrong twice over. The arm's own command pushes the
    hull back, and the hull is drifting under the water's forces regardless of what
    the arm does. This controller models both.

    Over one control step, with ``H`` the generalized mass matrix partitioned into
    the hull block ``H_b``, the coupling block ``H_bm`` and the arm block, momentum
    gives the hull's velocity change as::

        dv_b = H_b^-1 (w_ext dt - H_bm dq_dot)

    The first term is the drift the water imposes; the second is the reaction the
    arm imposes. The tool's displacement is then ``J_b dp_b + J_m dq``, and
    substituting turns it into::

        dx = J_b (v_b + dv_b) dt + (J_m - J_b H_b^-1 H_bm) dq

    The bracket is the generalized Jacobian of a free-floating manipulator: the map
    from a joint command to tool motion once the base's recoil is accounted for.
    Inverting it, against an error already corrected for the drift, is the whole
    controller.

    Two deliberate limitations, both about keeping the comparison honest:

    * ``w_ext`` is built from the *nominal* drag coefficients and the restoring
      term, never the simulation's. Under a drag mismatch this controller is wrong
      in exactly the way Stage 3 intends.
    * the wave force is not in ``w_ext`` at all. A vessel cannot measure the swell
      it is about to be hit by, and the policy is not told either; both see only its
      effect on the hull's measured state.

    ``H`` and ``J`` come from PhysX rather than from a hand-derived model. That is a
    real advantage over a shipboard controller, and it is granted deliberately: it
    makes baseline B the strongest opponent available, so that a policy beating it
    cannot be explained by the baseline having a sloppy rigid-body model.
    """

    #: How much of the step's hull velocity change lands inside the same step.
    #: Measured in `scripts/verify_coupling_model.py`, which predicts the hull's
    #: displacement at 0.5 and at 1.0 and finds 1.0 the better of the two. Fixed
    #: here before any controller ran, not tuned afterwards.
    REACTION_FACTOR = 1.0

    #: The two things this controller models beyond baseline A, separable so the gap
    #: between A and B can be attributed rather than asserted. ``drift`` feeds forward
    #: where the hull will have moved by the end of the step under the water's forces;
    #: ``reaction`` corrects the Jacobian for the hull's recoil from the arm's own
    #: command. ``both`` is the controller the plan specifies; the single-term modes
    #: exist only to measure which half is doing the work, and ``neither`` reduces this
    #: class to baseline A driven through the same code path, which is the control that
    #: proves the two are otherwise identical.
    TERM_MODES = ("both", "drift", "reaction", "neither")

    def __init__(
        self,
        env,
        gain: float = 1.0,
        damping: float = 0.05,
        terms: str = "both",
        oracle_drag: bool = False,
    ):
        """``oracle_drag`` hands this controller the drag the *simulation* is using.

        That breaks the plan's rule 2 on purpose, and it is never the headline baseline.
        It exists because the mismatch axis is confounded: multiplying the drag changes
        how wrong the controller's model is *and* how hard the task is, since a hull in
        thicker water simply moves less. Running the same controller with the true
        coefficients at each factor holds the difficulty fixed, so the gap between the
        oracle and the nominal version is the cost of the model error alone. It is a
        measuring instrument, labelled as one wherever it is reported.
        """
        from isaaclab.managers import SceneEntityCfg
        from marine_manipulator import calibration, hydrodynamics, ur3_kin

        if terms not in self.TERM_MODES:
            raise ValueError(f"terms must be one of {self.TERM_MODES}, got {terms!r}")
        self._oracle_drag = oracle_drag
        self._use_drift = terms in ("both", "drift")
        self._use_reaction = terms in ("both", "reaction")
        self.terms = terms
        self._env = env
        self._kin = ur3_kin
        self._hydro = hydrodynamics
        self._gain = gain
        self._damping = damping
        self._tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self._tool_cfg.resolve(env.scene)
        self._robot = env.scene["robot"]
        self._view = self._robot.root_physx_view
        self._term = env.action_manager._terms["arm_action"]
        self._params = mdp.vehicle_params(env)
        self._scale = calibration.REALISTIC_ACTION_SCALE_RAD
        self._offset_b = torch.tensor(calibration.TCP_OFFSET, device=env.device).repeat(
            env.num_envs, 1
        )
        self._wrist = self._robot.body_names.index("wrist_3_link")
        self._down = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
        self._joints = self._robot.data.joint_pos.shape[1]
        self.solved: list[float] = []

    def _external_wrench_w(self):
        """The hydrodynamic wrench the controller believes in, in world axes.

        Drag at nominal coefficients plus restoring. No wave term: it is not
        measurable in advance and the policy is not given it either.
        """
        linear, quadratic = self._params.linear_drag, self._params.quadratic_drag
        if self._oracle_drag:
            scale = mdp.drag_mismatch_scale(self._env).unsqueeze(1)
            linear = torch.tensor(linear, device=scale.device).unsqueeze(0) * scale
            quadratic = torch.tensor(quadratic, device=scale.device).unsqueeze(0) * scale
        drag_force, drag_torque = self._hydro.drag_wrench(
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            linear,
            quadratic,
        )
        restore_force, restore_torque = self._hydro.restoring_wrench(
            self._robot.data.root_quat_w,
            self._params.mass_kg,
            self._params.cob_offset_b,
            self._params.net_buoyancy_fraction,
        )
        quat = self._robot.data.root_quat_w
        return torch.cat(
            (
                quat_apply(quat, drag_force + restore_force),
                quat_apply(quat, drag_torque + restore_torque),
            ),
            dim=1,
        )

    def _task_error_w(self):
        """Six-vector the tool has to null: line offset, then tool-axis rotation."""
        position = mdp.last_measured_target_error(
            self._env, "ee_pose", self._tool_cfg, mdp.TCP_OFFSET
        )
        quat = self._robot.data.body_quat_w[:, self._wrist]
        current_axis = quat_apply(quat, self._offset_b)
        current_axis = current_axis / current_axis.norm(dim=1, keepdim=True)
        axis = torch.cross(current_axis, self._down, dim=1)
        sine = torch.linalg.norm(axis, dim=1, keepdim=True)
        cosine = (current_axis * self._down).sum(dim=1, keepdim=True)
        rotation = axis / sine.clamp(min=1e-8) * torch.atan2(sine, cosine)
        return torch.cat((position, rotation), dim=1)

    def __call__(self, obs):
        data = self._robot.data
        dt = self._env.step_dt

        matrices = self._view.get_generalized_mass_matrices()
        h_b = matrices[:, :6, :6]
        h_bm = matrices[:, :6, 6:]

        # PhysX reports the Jacobian of the link's own frame; the tool tip sits
        # 0.12 m along it, so the linear rows have to be shifted by that lever arm
        # or the tool's rotation would contribute nothing to its translation.
        jac = self._view.get_jacobians()[:, self._wrist]
        lever = quat_apply(data.body_quat_w[:, self._wrist], self._offset_b)
        jac = torch.cat((jac[:, :3] - _skew(lever) @ jac[:, 3:], jac[:, 3:]), dim=1)
        jac_b, jac_m = jac[:, :, :6], jac[:, :, 6:]

        if self._use_drift:
            wrench = self._external_wrench_w()
            velocity = torch.cat((data.root_lin_vel_w, data.root_ang_vel_w), dim=1)
            hull_delta = torch.linalg.solve(h_b, wrench.unsqueeze(2) * dt).squeeze(2)
            drift = (
                jac_b @ ((velocity + self.REACTION_FACTOR * hull_delta) * dt).unsqueeze(2)
            ).squeeze(2)
        else:
            drift = torch.zeros((data.root_lin_vel_w.shape[0], 6), device=jac.device)

        if self._use_reaction:
            generalized = jac_m - self.REACTION_FACTOR * (
                jac_b @ torch.linalg.solve(h_b, h_bm)
            )
        else:
            generalized = jac_m
        error = self._task_error_w() * self._gain - drift

        identity = torch.eye(6, device=jac.device, dtype=jac.dtype).expand_as(generalized)
        damped = generalized @ generalized.transpose(1, 2) + (self._damping**2) * identity
        delta = generalized.transpose(1, 2) @ torch.linalg.solve(damped, error.unsqueeze(2))
        command = data.joint_pos + delta.squeeze(2)
        return (command - self._term._offset) / self._scale


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    """Batched cross-product matrices, ``(N, 3) -> (N, 3, 3)``."""
    zero = torch.zeros_like(vectors[:, 0])
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=1),
            torch.stack((z, zero, -x), dim=1),
            torch.stack((-y, x, zero), dim=1),
        ),
        dim=1,
    )
