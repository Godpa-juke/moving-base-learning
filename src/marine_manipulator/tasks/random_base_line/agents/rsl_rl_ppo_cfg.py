from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.ur3.agents.rsl_rl_ppo_cfg import (
    UR3ReachPPORunnerCfg,
)


@configclass
class RandomBaseWorldLinePPORunnerCfg(UR3ReachPPORunnerCfg):
    experiment_name = "marine_random_6dof_base_worldline"
    max_iterations = 5000
    save_interval = 50


@configclass
class IkSeededWorldLinePPORunnerCfg(RandomBaseWorldLinePPORunnerCfg):
    """Bounds the action the policy may emit.

    The runner default is ``clip_actions = None``, so nothing limited the raw output
    except the environment's rate limiter, which the action manager never saw.
    """

    experiment_name = "marine_ik_seeded_worldline"
    clip_actions = 3.0
    max_iterations = 10000


@configclass
class SensorDegradedWorldLinePPORunnerCfg(IkSeededWorldLinePPORunnerCfg):
    """Wider network: the measured-base-state history makes the observation ~250-d,
    which the stock 64x64 reach network is too small to exploit."""

    experiment_name = "marine_sensor_degraded_worldline"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )


@configclass
class UvmsWorldLinePPORunnerCfg(IkSeededWorldLinePPORunnerCfg):
    """Free-floating round.

    The observation is the same size as the IK-seeded task's, but the mapping from
    action to tool motion now runs through the hull's dynamics rather than through the
    arm alone, which is a harder function than a 64x64 network fits comfortably. The
    wider network is inherited from the degraded task for that reason, not because the
    observation grew.
    """

    experiment_name = "marine_uvms_worldline"
    max_iterations = 10000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )


@configclass
class ResidualIkRecurrentPPORunnerCfg(IkSeededWorldLinePPORunnerCfg):
    """Recurrent policy for the residual task.

    Actuation delay makes the environment genuinely partially observable: the arm's
    current configuration reflects a command issued up to three steps ago, and no single
    frame distinguishes "the target moved" from "my last command has not landed yet".
    Herland & Bach use a recurrent policy for the same reason. Both previous rounds used
    a feedforward network over a stacked observation window, which is the weaker
    substitute and was never compared against recurrence.
    """

    experiment_name = "marine_residual_ik_delay"
    max_iterations = 6000
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
    )
