from __future__ import annotations

import math

from isaaclab.envs import mdp as isaac_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR3_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.reach.config.ur3.joint_pos_env_cfg import (
    UR3ReachEnvCfg,
    UR3ReachEnvCfg_PLAY,
)

from marine_manipulator import calibration as mdp_calibration

from . import mdp


@configclass
class RandomBaseWorldLineEnvCfg(UR3ReachEnvCfg):
    """Fresh stock-UR3 baseline with episode-random external 6-DoF base motion."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UR3_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5

        tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self.commands.ee_pose.class_type = mdp.WorldHorizontalLineCommand
        self.commands.ee_pose.body_name = "wrist_3_link"
        self.commands.ee_pose.body_offset = mdp.TCP_OFFSET
        self.commands.ee_pose.ranges.pos_x = (0.22, 0.38)
        self.commands.ee_pose.ranges.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges.pos_z = (0.20, 0.36)
        self.commands.ee_pose.ranges.roll = (0.0, 0.0)
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi)
        self.commands.ee_pose.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.ee_pose.amplitude_range = (0.08, 0.14)
        self.commands.ee_pose.period_range = (5.0, 8.0)

        self.events.sample_base_motion = EventTerm(func=mdp.sample_base_motion, mode="reset")
        self.events.apply_base_motion = EventTerm(
            func=mdp.apply_base_motion,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.observations.policy.base_pose = ObsTerm(func=mdp.normalized_base_pose)
        self.observations.policy.base_velocity = ObsTerm(func=mdp.normalized_base_velocity)
        self.observations.policy.tcp_position_w = ObsTerm(
            func=mdp.tcp_position_w,
            params={
                "asset_cfg": tool_cfg, "body_offset": mdp.TCP_OFFSET,
              },
          )
        self.observations.policy.enable_corruption = False

        for name in (
             "end_effector_position_tracking",
             "end_effector_position_tracking_fine_grained",
             "end_effector_orientation_tracking",
         ):
            if hasattr(self.rewards, name):
                delattr(self.rewards, name)
        self.rewards.position = RewTerm(
            func=mdp.position_error,
            weight=-0.08,
            params={
                 "asset_cfg": tool_cfg,
                 "command_name": "ee_pose",
                 "body_offset": mdp.TCP_OFFSET,
              },
         )
        self.rewards.position_fine = RewTerm(
            func=mdp.position_error_tanh,
            weight=0.35,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "std": 0.06,
            },
        )
        self.rewards.orientation = RewTerm(
            func=mdp.orientation_error,
            weight=-0.02,
            params={"asset_cfg": tool_cfg, "command_name": "ee_pose",
            },
         )
        self.rewards.action_rate = RewTerm(func=isaac_mdp.action_rate_l2, weight=-0.00005)
        self.rewards.joint_vel = RewTerm(
            func=isaac_mdp.joint_vel_l2,
            weight=-0.00005,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
             },
         )


@configclass
class PrecisionStartWorldLineEnvCfg(RandomBaseWorldLineEnvCfg):
    """One-centimeter task: aligned -Y endpoint start, hold, then traverse."""

    def __post_init__(self):
        super().__post_init__()
        tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self.scene.robot.init_state.joint_pos = {
            name: position for name, position in zip(
                (
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "elbow_joint",
                    "wrist_1_joint",
                    "wrist_2_joint",
                    "wrist_3_joint",
                ),
                mdp.PRECISION_START_JOINT_POS,
            )
        }
        self.commands.ee_pose.class_type = mdp.PrecisionStartLineCommand
        start_x, start_y, start_z = mdp.PRECISION_START_TCP_E
        self.commands.ee_pose.ranges.pos_x = (start_x, start_x)
        line_center_y = start_y + 0.10
        self.commands.ee_pose.ranges.pos_y = (line_center_y, line_center_y)
        self.commands.ee_pose.ranges.pos_z = (start_z, start_z)
        self.commands.ee_pose.ranges.yaw = (0.0, 0.0)
        self.commands.ee_pose.amplitude_range = (0.10, 0.10)
        self.commands.ee_pose.period_range = (8.0, 8.0)
        self.commands.ee_pose.hold_duration_s = 1.0
        self.commands.ee_pose.traverse_duration_s = 4.0

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_start_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.events.sample_base_motion = EventTerm(
            func=mdp.sample_precision_base_motion,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # PrecisionStartRLEnv applies the sampled root motion before physics.
        # Keeping the inherited interval event would invalidate link/TCP buffers
        # between reward and observation computation.
        self.events.apply_base_motion = None
        self.observations.policy.base_pose = ObsTerm(func=mdp.normalized_neutral_start_base_pose)
        self.observations.policy.base_velocity = ObsTerm(
            func=mdp.normalized_neutral_start_base_velocity
        )

        self.rewards.position_precision = RewTerm(
            func=mdp.position_error_tanh,
            weight=0.75,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "std": 0.015,
            },
        )
        self.rewards.success_1cm = RewTerm(
            func=mdp.position_success,
            weight=0.50,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "threshold": 0.01,
            },
        )
        self.rewards.orientation = None


@configclass
class TargetConditionedWorldLineEnvCfg(PrecisionStartWorldLineEnvCfg):
    """Command-conditioned tracker trained across line locations, not one memorized path."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4096
        tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self.commands.ee_pose.class_type = mdp.ReachableRandomLineCommand
        self.commands.ee_pose.ranges.pos_x = (-0.40, -0.30)
        self.commands.ee_pose.ranges.pos_y = (-0.03, 0.03)
        self.commands.ee_pose.ranges.pos_z = (0.18, 0.24)
        self.commands.ee_pose.line_direction_e = (0.0, 1.0, 0.0)
        self.commands.ee_pose.amplitude_range = (0.05, 0.07)
        self.commands.ee_pose.hold_duration_s = 3.0
        self.commands.ee_pose.traverse_duration_s = 6.0
        self.commands.ee_pose.reachable_radius_m = 0.50
        self.observations.policy.base_pose = ObsTerm(func=mdp.normalized_immediate_base_pose)
        self.observations.policy.base_velocity = ObsTerm(
            func=mdp.normalized_immediate_base_velocity
        )
        self.observations.policy.tcp_position_w = None
        self.observations.policy.target_error_w = ObsTerm(
            func=mdp.target_error_w,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
            },
        )
        self.rewards.position_coarse = RewTerm(
            func=mdp.position_error_tanh,
            weight=1.0,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "std": 0.15,
            },
        )
        self.rewards.position_medium = RewTerm(
            func=mdp.position_error_tanh,
            weight=0.8,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "std": 0.05,
            },
        )
        self.rewards.tcp_speed_limit = RewTerm(
            func=mdp.tcp_speed_limit_penalty,
            weight=-4.0,
            params={
                "asset_cfg": tool_cfg,
                "body_offset": mdp.TCP_OFFSET,
                "max_speed_m_s": 0.15,
            },
        )
        self.rewards.action_rate.weight = -0.01
        self.rewards.joint_vel.weight = -0.001


@configclass
class IkSeededWorldLineEnvCfg(TargetConditionedWorldLineEnvCfg):
    """Line tracking with an IK-placed start, a constrained tool axis, and honest
    action penalties.

    Three changes relative to the parent, each aimed at one measured defect:

    1. The episode starts on the line (analytic IK, cylinder vertical) instead of
       0.65 m away, so the reported error is tracking error rather than a average
       over a long transit.
    2. The cylinder axis is rewarded for pointing straight down. The parent leaves
       orientation free and drifts to ~1.7 rad, which lets the tip reach the line by
       pivoting the tool rather than following it.
    3. Penalties act on the policy's raw output. The parent penalises the
       rate-limited signal, leaving the raw output unconstrained; the actor mean
       then grows until the limiter saturates and tracking decays over training.
    """

    def __post_init__(self):
        super().__post_init__()
        tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        joint_names = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        # The joint-position action is `default_joint_pos + 0.5 * action`, so the
        # default has to sit at the centre of the workspace or the policy spends its
        # action range just getting there.
        self.scene.robot.init_state.joint_pos = dict(
            zip(joint_names, mdp_calibration.BOX_CENTER_JOINT_POS)
        )
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_box_center,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.tool_axis = RewTerm(
            func=mdp.tool_axis_alignment,
            weight=0.5,
            params={"asset_cfg": tool_cfg, "std": 0.15},
        )
        self.rewards.position_mm = RewTerm(
            func=mdp.position_error_tanh,
            weight=0.5,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "std": 0.003,
            },
        )
        self.rewards.success_2mm = RewTerm(
            func=mdp.position_success,
            weight=0.5,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
                "threshold": 0.002,
            },
        )

        # Swap the limited-signal penalties for raw-output ones. The curriculum term
        # that rewrites `action_rate`'s weight has to go with it.
        self.rewards.action_rate = None
        self.curriculum.action_rate = None
        self.rewards.raw_action_l2 = RewTerm(func=mdp.raw_action_l2, weight=-0.002)
        self.rewards.raw_action_rate = RewTerm(func=mdp.raw_action_rate_l2, weight=-0.01)


@configclass
class IkSeededWorldLineEnvCfg_PLAY(IkSeededWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SensorDegradedWorldLineEnvCfg(IkSeededWorldLineEnvCfg):
    """The IK-seeded task with a realistic base-motion sensor.

    The baseline hands the policy the exact disturbance, analytically evaluated from
    the amplitude/frequency/phase it was sampled with, and hands the model-based
    controller the true root pose. A vessel offers neither. Here both read one shared
    measurement that is late by up to four control steps and carries additive noise,
    with the amount randomised per episode so that a single policy can be evaluated
    across the whole delay sweep.

    The policy is given a window of past measurements: predicting through the delay
    means estimating the phase and frequency of the disturbance, which a single frame
    cannot support.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.sample_sensor_degradation = EventTerm(
            func=mdp.sample_sensor_degradation,
            mode="reset",
            params={
                "delay_range_s": mdp_calibration.SENSOR_DELAY_RANGE_S,
                "position_noise_range_m": mdp_calibration.SENSOR_POSITION_NOISE_RANGE_M,
                "rotation_noise_range_rad": mdp_calibration.SENSOR_ROTATION_NOISE_RANGE_RAD,
            },
        )
        history = mdp_calibration.SENSOR_HISTORY_LENGTH
        self.observations.policy.base_pose = ObsTerm(
            func=mdp.normalized_measured_base_pose, history_length=history
        )
        self.observations.policy.base_velocity = ObsTerm(
            func=mdp.normalized_measured_base_velocity, history_length=history
        )


@configclass
class SensorDegradedWorldLineEnvCfg_PLAY(SensorDegradedWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SensorDegradedSeamEnvCfg(SensorDegradedWorldLineEnvCfg):
    """Degrade the seam measurement as well as the base measurement.

    Degrading only the base measurement changed nothing, because `target_error_w` still
    handed the policy the exact vector from its true TCP to the true target: with that
    it can close the loop on error and never needs to know the base moved. A seam
    tracker reports the same offset late and noisily, so here both channels are degraded
    on the same terms and the model-based controller reads the same two signals.

    This is the condition under which predicting the disturbance is actually necessary.
    """

    def __post_init__(self):
        super().__post_init__()
        tool_cfg = SceneEntityCfg("robot", body_names=["wrist_3_link"])
        self.observations.policy.target_error_w = ObsTerm(
            func=mdp.measured_target_error_w,
            params={
                "asset_cfg": tool_cfg,
                "command_name": "ee_pose",
                "body_offset": mdp.TCP_OFFSET,
            },
            history_length=mdp_calibration.SENSOR_HISTORY_LENGTH,
        )


@configclass
class SensorDegradedSeamEnvCfg_PLAY(SensorDegradedSeamEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SensorDegradedNoHistoryEnvCfg(SensorDegradedWorldLineEnvCfg):
    """Ablation: identical degradation, but the policy sees one frame instead of many.

    Isolates the mechanism. If the full task's delay robustness came from estimating
    the disturbance phase over a window, this loses it; if it came from retraining
    under degradation alone, this keeps it. Without the ablation, a crossover against
    the analytic controller is not attributable to anything in particular.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.base_pose = ObsTerm(
            func=mdp.normalized_measured_base_pose, history_length=1
        )
        self.observations.policy.base_velocity = ObsTerm(
            func=mdp.normalized_measured_base_velocity, history_length=1
        )


@configclass
class SensorDegradedWideDelayEnvCfg(SensorDegradedWorldLineEnvCfg):
    """Degradation with the delay range widened to a third of a second.

    Held in reserve for the case where the analytic controller stays competitive
    across the whole baseline range: the comparison then needs a delay large enough
    to separate a reactive controller from a predictive one, and the policy has to be
    trained over that range rather than extrapolated into it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.sample_sensor_degradation.params["delay_range_s"] = (0.0, 0.3333)


@configclass
class SensorDegradedNoHistoryEnvCfg_PLAY(SensorDegradedNoHistoryEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class SensorDegradedWideDelayEnvCfg_PLAY(SensorDegradedWideDelayEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class TargetCondWorldLineSuccessBiasEnvCfg(TargetConditionedWorldLineEnvCfg):
    """Success-biased continuation of TargetConditionedWorldLine (independent experiment).

    Same geometry / observation / rate limiter / speed cap semantics as the parent,
    but boosts the two active tracking kernels and the 1cm success bonus to break
    the 5cm -> 1cm plateau observed in the 4096-env baseline. All other reward
    terms are inherited unchanged, preserving the wide pursuit that already works.
    This is a *new* task ID so we resume-from-checkpoint without altering the
    baseline that produced model_9999.pt.
    """

    def __post_init__(self):
        super().__post_init__()
        # Boost the active medium/precision position kernels (std 0.05 / 0.015,
        # inherited from the precision parent stack) so the 5-1cm band receives
        # materially more gradient. Coarse (std 0.15) stays at 1.0 to keep pursuit.
        self.rewards.position_medium.weight = 1.5
        self.rewards.position_precision.weight = 1.4
        self.rewards.success_1cm.weight = 1.5
        # Relax the overspeed penalty: a 0.15 m/s cap with weight -4.0 can force the
        # policy to crawl and stall the final centimeter. -2.0 keeps it bounded but
        # allows faster terminal approach.
        self.rewards.tcp_speed_limit.weight = -2.0


@configclass
class TargetCondWorldLineSuccessBiasEnvCfg_PLAY(
    TargetCondWorldLineSuccessBiasEnvCfg, UR3ReachEnvCfg_PLAY
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class PrecisionFarImmediateBaseEnvCfg(PrecisionStartWorldLineEnvCfg):
    """Precision line shifted -0.50 m on world X; base moves from the first step."""

    def __post_init__(self):
        super().__post_init__()
        joint_names = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        self.scene.robot.init_state.joint_pos = dict(zip(joint_names, mdp.PRECISION_FAR_START_JOINT_POS))
        start_x, start_y, start_z = mdp.PRECISION_FAR_START_TCP_E
        self.commands.ee_pose.ranges.pos_x = (start_x, start_x)
        self.commands.ee_pose.ranges.pos_y = (start_y + 0.10, start_y + 0.10)
        self.commands.ee_pose.ranges.pos_z = (start_z, start_z)
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_far_start_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.observations.policy.base_pose = ObsTerm(func=mdp.normalized_immediate_base_pose)
        self.observations.policy.base_velocity = ObsTerm(
            func=mdp.normalized_immediate_base_velocity
        )


@configclass
class TargetConditionedWorldLineEnvCfg_PLAY(TargetConditionedWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class PrecisionFarImmediateBaseEnvCfg_PLAY(PrecisionFarImmediateBaseEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class PrecisionStartWorldLineEnvCfg_PLAY(PrecisionStartWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class RandomBaseWorldLineEnvCfg_PLAY(RandomBaseWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class UvmsWorldLineEnvCfg(IkSeededWorldLineEnvCfg):
    """The IK-seeded line task with the vehicle released into the water.

    Four differences from the parent, and only four:

    1. The articulation root moves off the world fixed joint onto ``base_link``, which
       carries the vehicle's mass and inertia. ``uvms_asset`` explains why the root has
       to move rather than simply be unfixed.
    2. The prescribed root motion is gone. ``apply_immediate_base_motion`` teleported
       the base every step, which is precisely what made the arm unable to influence
       it; :class:`UvmsLineRLEnv` applies a hydrodynamic wrench instead and lets PhysX
       produce the arm's reaction.
    3. The base observation reads the hull's *actual* pose and velocity rather than the
       analytic sinusoid, because there is no longer a sinusoid to evaluate.
    4. The table and the ground plane are removed. Both are furniture from a bench-top
       reach task; a hull free to drift can hit them, and a collision would be
       indistinguishable in the metrics from a control failure.

    The drag mismatch defaults to none. The mismatch-randomised variants below are what
    Stage 3 and Stage 4 use.
    """

    #: Vehicle dry mass, kg. ``None`` takes the nominal one from ``calibration.py``.
    #: The Stage 0 sweep varies this; nothing else should.
    vehicle_mass_kg: float | None = None

    def __post_init__(self):
        super().__post_init__()
        from marine_manipulator import uvms_asset

        params = (
            uvms_asset.default_vehicle_params()
            if self.vehicle_mass_kg is None
            else uvms_asset.scaled_vehicle_params(self.vehicle_mass_kg)
        )
        joint_pos = self.scene.robot.init_state.joint_pos
        robot = uvms_asset.uvms_robot_cfg(params)
        robot.prim_path = "{ENV_REGEX_NS}/Robot"
        robot.init_state = robot.init_state.replace(joint_pos=joint_pos)
        self.scene.robot = robot
        self.scene.table = None
        self.scene.ground = None

        # The wave amplitude/frequency/phase draw is unchanged; only what is done with
        # it changes, from a prescribed pose to a force. Keeping the same distribution
        # is what makes the two rounds' disturbances comparable.
        self.events.sample_base_motion = EventTerm(func=mdp.sample_base_motion, mode="reset")
        self.events.reset_uvms_root = EventTerm(
            func=mdp.reset_uvms_root,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.events.apply_base_motion = None

        self.observations.policy.base_pose = ObsTerm(func=mdp.normalized_uvms_base_pose)
        self.observations.policy.base_velocity = ObsTerm(func=mdp.normalized_uvms_base_velocity)


@configclass
class UvmsWorldLineEnvCfg_PLAY(UvmsWorldLineEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


@configclass
class UvmsDragRandomizedEnvCfg(UvmsWorldLineEnvCfg):
    """Domain randomisation over the drag the *simulation* uses.

    The controllers and the policy keep the nominal coefficients from
    ``calibration.py``; this event makes the water disagree with them, by a factor drawn
    log-uniformly over the training range. It is what Stage 4 trains against and what
    Stage 3 sweeps by pinning the factor.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.sample_drag_mismatch = EventTerm(
            func=mdp.sample_drag_mismatch,
            mode="reset",
            params={"scale_range": mdp_calibration.DRAG_MISMATCH_TRAIN_RANGE},
        )


@configclass
class UvmsDragRandomizedEnvCfg_PLAY(UvmsDragRandomizedEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False


# Stage 0's mass sweep. The plan asks for {50, 100, 200, 500, 2000} kg; a work-class
# 2000 kg hull is included so the trend has an end the plan already calls dead, which is
# what makes the shape of the curve interpretable rather than just its endpoint.
_STAGE0_MASSES = (50.0, 200.0, 500.0, 2000.0)


def _mass_variant(mass_kg: float):
    """Build a ``UvmsWorldLineEnvCfg`` subclass carrying one vehicle mass."""

    @configclass
    class _Variant(UvmsWorldLineEnvCfg):
        vehicle_mass_kg: float | None = mass_kg

    @configclass
    class _VariantPlay(_Variant, UR3ReachEnvCfg_PLAY):
        def __post_init__(self):
            super().__post_init__()
            self.scene.num_envs = 16
            self.observations.policy.enable_corruption = False

    return _Variant, _VariantPlay


for _mass in _STAGE0_MASSES:
    _name = f"UvmsWorldLine{int(_mass)}kgEnvCfg"
    globals()[_name], globals()[f"{_name}_PLAY"] = _mass_variant(_mass)


@configclass
class ResidualIkDelayEnvCfg(IkSeededWorldLineEnvCfg):
    """Residual policy on top of the analytic controller, under actuation delay.

    The condition finding 17 identified as the one where a policy has anything to win:
    the delay sits inside the feedback loop, where it costs the analytic controller
    1.6-2.1x, rather than on the measurement, where it costs nothing.

    Both the delay and the disturbance speed are set per episode by an event, so one
    policy covers the range the evaluation sweeps rather than needing a separate agent
    per cell. Herland & Bach train one agent per delay on the grounds that the delay is a
    fixed property of the arm; randomising is the harder version and it also avoids
    reporting five agents' best cells as though they were one agent's curve.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.sample_actuation_conditions = EventTerm(
            func=mdp.sample_actuation_conditions,
            mode="reset",
            params={
                "delay_steps_range": (0, 3),
                "frequency_scale_range": (1.0, 4.0),
            },
        )
        # The policy must know how stale its commands are; the delay is not otherwise
        # inferable from a single frame, and Herland & Bach sidestep this by training one
        # agent per delay. Recurrence over the observation history is what has to
        # substitute for that, so the network is recurrent (see the runner config).
        self.observations.policy.actuation_conditions = ObsTerm(
            func=mdp.normalized_actuation_conditions
        )


@configclass
class ResidualIkDelayEnvCfg_PLAY(ResidualIkDelayEnvCfg, UR3ReachEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
