from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import mdp


class PrecisionStartRLEnv(ManagerBasedRLEnv):
    """Apply the externally driven base before physics, not after rewards."""

    def step(self, action: torch.Tensor):
        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            mdp.reset_joints_to_start_pose(self, reset_ids)
        mdp.apply_neutral_start_base_motion(self, None)
        return super().step(action)


class TargetConditionedRLEnv(ManagerBasedRLEnv):
    """Move the randomized base immediately while the policy follows the live command."""

    def step(self, action: torch.Tensor):
        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if not hasattr(self, "_previous_rate_limited_action"):
            self._previous_rate_limited_action = torch.zeros_like(action)
        if reset_ids.numel() > 0:
            mdp.reset_joints_to_start_pose(self, reset_ids)
            self._previous_rate_limited_action[reset_ids] = 0.0
        max_speed = torch.tensor(mdp.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=action.device)
        limited_action = mdp.rate_limit_joint_action(
            self._previous_rate_limited_action,
            action,
            max_speed_rad_s=max_speed,
            step_dt=self.step_dt,
            action_scale=mdp.REALISTIC_ACTION_SCALE_RAD,
        )
        self._previous_rate_limited_action = limited_action.detach().clone()
        mdp.apply_immediate_base_motion(self, None)
        return super().step(limited_action)


class PrecisionFarImmediateBaseRLEnv(ManagerBasedRLEnv):
    """Far-line precision task with base motion beginning on the first step."""

    def step(self, action: torch.Tensor):
        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            mdp.reset_joints_to_far_start_pose(self, reset_ids)
        mdp.apply_immediate_base_motion(self, None)
        return super().step(action)


class IkSeededLineRLEnv(ManagerBasedRLEnv):
    """Target-conditioned tracker that starts every episode already on the line.

    Differences from :class:`TargetConditionedRLEnv`:

    * the arm is placed by analytic IK on the sampled line's start point with the
      cylinder vertical, so an episode contains no 0.65 m transit to average into
      the tracking metric;
    * the policy's own output is stashed before rate limiting so reward terms can
      penalise it, closing the loophole that let the actor mean drift unbounded.
    """

    def step(self, action: torch.Tensor):
        # Must happen for every task built on this env, not just the degraded one:
        # a stale measurement cache silently freezes the base state a controller
        # reads for the whole episode, which looks like a badly tuned controller
        # rather than a bug.
        mdp.clear_sensor_cache(self)
        mdp.clear_seam_cache(self)
        previous_raw = getattr(self, mdp._RAW_ACTION_ATTR, None)
        setattr(self, mdp._PREV_RAW_ACTION_ATTR, previous_raw if previous_raw is not None else action.clone())
        setattr(self, mdp._RAW_ACTION_ATTR, action.clone())

        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if not hasattr(self, "_previous_rate_limited_action"):
            self._previous_rate_limited_action = torch.zeros_like(action)
        if reset_ids.numel() > 0:
            # Commands are resampled after reset events, so the IK seed has to happen
            # here, where the line for this episode is finally known.
            mdp.reset_joints_to_ik_line_start(self, reset_ids)
            mdp.reset_seam_buffer(self, reset_ids)
            self._previous_rate_limited_action[reset_ids] = 0.0
            setattr(self, mdp._PREV_RAW_ACTION_ATTR, getattr(self, mdp._RAW_ACTION_ATTR))

        max_speed = torch.tensor(mdp.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=action.device)
        limited_action = mdp.rate_limit_joint_action(
            self._previous_rate_limited_action,
            action,
            max_speed_rad_s=max_speed,
            step_dt=self.step_dt,
            action_scale=mdp.REALISTIC_ACTION_SCALE_RAD,
        )
        self._previous_rate_limited_action = limited_action.detach().clone()
        mdp.apply_immediate_base_motion(self, None)
        # The rate limiter is the controller's own state and runs on what was commanded;
        # the delay is physical and applies to what the arm receives. Limiting first and
        # delaying second is therefore the correct order.
        return super().step(mdp.delay_actuation(self, limited_action))


class SensorDegradedLineRLEnv(IkSeededLineRLEnv):
    """Same physics as :class:`IkSeededLineRLEnv`, but the base state is only
    *measured* through a delayed, noisy sensor.

    The disturbance itself is unchanged; what changes is that neither the policy nor
    the model-based baseline can see it exactly any more. Behaviour comes entirely
    from the configuration, so this exists to give the task its own entry point and
    to document the difference.
    """


class UvmsLineRLEnv(IkSeededLineRLEnv):
    """Line tracking with the vehicle free to move, and pushed by the arm.

    The parent teleports the root every step through ``apply_immediate_base_motion``,
    which makes the base an exogenous signal and the task a pure inversion. Here the
    root is a free body: the only things acting on it are the hydrodynamic wrench
    applied below and the reaction PhysX transmits from the arm's own motion. That
    reaction is the point of the round and needs no code of its own.

    Everything else the parent does is kept deliberately - the IK start seeding, the
    joint rate limiter, the raw-action penalties and the action clipping. Those fixes
    address defects that have nothing to do with how the base moves (``FINDINGS.md``
    sections 3 and 4) and are still needed.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # The wrench terms and the coupled controller both ask the environment which
        # vehicle it is carrying. Installing it here, from the same configuration that
        # sized the articulation's mass, is what stops the two from disagreeing.
        mass = getattr(cfg, "vehicle_mass_kg", None)
        if mass is not None:
            from marine_manipulator import uvms_asset

            setattr(self, mdp._VEHICLE_ATTR, uvms_asset.scaled_vehicle_params(mass))

    def step(self, action: torch.Tensor):
        mdp.clear_sensor_cache(self)
        mdp.clear_seam_cache(self)
        setattr(self, mdp._FREE_FLOATING_ATTR, True)
        previous_raw = getattr(self, mdp._RAW_ACTION_ATTR, None)
        setattr(self, mdp._PREV_RAW_ACTION_ATTR, previous_raw if previous_raw is not None else action.clone())
        setattr(self, mdp._RAW_ACTION_ATTR, action.clone())

        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if not hasattr(self, "_previous_rate_limited_action"):
            self._previous_rate_limited_action = torch.zeros_like(action)
        if reset_ids.numel() > 0:
            mdp.reset_joints_to_ik_line_start(self, reset_ids)
            mdp.reset_seam_buffer(self, reset_ids)
            self._previous_rate_limited_action[reset_ids] = 0.0
            setattr(self, mdp._PREV_RAW_ACTION_ATTR, getattr(self, mdp._RAW_ACTION_ATTR))

        max_speed = torch.tensor(mdp.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=action.device)
        limited_action = mdp.rate_limit_joint_action(
            self._previous_rate_limited_action,
            action,
            max_speed_rad_s=max_speed,
            step_dt=self.step_dt,
            action_scale=mdp.REALISTIC_ACTION_SCALE_RAD,
        )
        self._previous_rate_limited_action = limited_action.detach().clone()
        mdp.apply_hydrodynamic_wrench(self)
        return super(IkSeededLineRLEnv, self).step(mdp.delay_actuation(self, limited_action))


class ResidualIkLineRLEnv(IkSeededLineRLEnv):
    """The analytic controller drives the arm; the policy nudges what it is aiming at.

    Both previous rounds replaced the controller outright, so the policy had to
    rediscover Jacobian inversion before it could improve on anything, and it never got
    past reproducing the naive baseline (``docs/FINDINGS.md`` 16). Residual policy
    learning (Silver et al. 2018; Johannink et al. 2019) instead keeps the conventional
    controller and learns a small correction on top, which is the architecture Herland &
    Bach (2023) use to get 43-89% on very nearly this task.

    The correction is applied to the controller's **input** - the task-space error it is
    trying to null - rather than to its joint output. That is what those papers do, and
    it keeps the correction in metres and radians, where a bound on it means something
    physical, instead of in joint angles where the same bound means different things in
    different configurations.

    The action is therefore reinterpreted: six numbers that were joint targets are now
    ``[dx, dy, dz, drx, dry, drz]`` in the root frame, scaled by
    :attr:`RESIDUAL_SCALE`. The dimension is unchanged, so the action term and the
    existing raw-action penalties carry over - and those penalties now do exactly what
    residual learning wants, since they penalise the correction rather than the command.
    """

    #: **Known deviation from canonical RPL.** Silver et al. initialise the residual
    #: network's last layer to zero, so the correction starts at exactly nothing and
    #: "should never make a good initial policy worse"; they also warn that a good
    #: initial policy paired with an untrained critic degrades early. This
    #: implementation does neither - the policy opens by emitting corrections of order
    #: ``init_noise_std * RESIDUAL_SCALE`` ~ 2.5 mm, comparable to the error it is meant
    #: to fix, and has to climb back. Seed 970's curve is consistent with paying that
    #: cost: it opened at 2.52 mm and descended to 1.37. Fix this before concluding
    #: anything negative about residual learning here.
    #:
    #: Metres (and radians) of task-space correction per unit of policy output. With the
    #: runner's ``clip_actions = 3.0`` the correction is bounded at 15 mm, which is
    #: several times the error the controller leaves under actuation delay and small
    #: enough that the policy cannot simply take over. A choice, not a measurement:
    #: sweep it if the results are marginal rather than assuming it is right.
    RESIDUAL_SCALE = 0.005

    def _residual_controller(self):
        from marine_manipulator.controllers import AnalyticIkController

        controller = getattr(self, "_marine_residual_controller", None)
        if controller is None:
            controller = AnalyticIkController(self, iterations=2, mode="seam", gain=1.0)
            self._marine_residual_controller = controller
        return controller

    def step(self, action: torch.Tensor):
        mdp.clear_sensor_cache(self)
        mdp.clear_seam_cache(self)
        previous_raw = getattr(self, mdp._RAW_ACTION_ATTR, None)
        setattr(self, mdp._PREV_RAW_ACTION_ATTR, previous_raw if previous_raw is not None else action.clone())
        setattr(self, mdp._RAW_ACTION_ATTR, action.clone())

        reset_ids = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if not hasattr(self, "_previous_rate_limited_action"):
            self._previous_rate_limited_action = torch.zeros_like(action)
        if reset_ids.numel() > 0:
            mdp.reset_joints_to_ik_line_start(self, reset_ids)
            mdp.reset_seam_buffer(self, reset_ids)
            self._previous_rate_limited_action[reset_ids] = 0.0
            setattr(self, mdp._PREV_RAW_ACTION_ATTR, getattr(self, mdp._RAW_ACTION_ATTR))

        # The controller reads this on its way past; it is cleared afterwards so an
        # evaluation running the same controller without a policy is unaffected.
        self._marine_task_residual = action * self.RESIDUAL_SCALE
        try:
            command = self._residual_controller()(None)
        finally:
            self._marine_task_residual = None

        max_speed = torch.tensor(mdp.REALISTIC_JOINT_SPEED_LIMIT_RAD_S, device=action.device)
        limited_action = mdp.rate_limit_joint_action(
            self._previous_rate_limited_action,
            command,
            max_speed_rad_s=max_speed,
            step_dt=self.step_dt,
            action_scale=mdp.REALISTIC_ACTION_SCALE_RAD,
        )
        self._previous_rate_limited_action = limited_action.detach().clone()
        mdp.apply_immediate_base_motion(self, None)
        return super(IkSeededLineRLEnv, self).step(mdp.delay_actuation(self, limited_action))
