from __future__ import annotations

import gymnasium as gym

from . import agents

TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLine-v0"
PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLine-Play-v0"
PRECISION_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-v0"
PRECISION_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionStart-Play-v0"
PRECISION_FAR_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionFar50cm-v0"
PRECISION_FAR_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLinePrecisionFar50cm-Play-v0"
TARGET_CONDITIONED_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineTargetConditioned-v0"
TARGET_CONDITIONED_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineTargetConditioned-Play-v0"
SUCCESS_BIAS_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineTargetCondSuccessBias-v0"
SUCCESS_BIAS_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineTargetCondSuccessBias-Play-v0"
IK_SEEDED_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineIkSeeded-v0"
IK_SEEDED_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineIkSeeded-Play-v0"
SENSOR_DEGRADED_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-v0"
SENSOR_DEGRADED_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineSensorDegraded-Play-v0"
NO_HISTORY_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedNoHistory-v0"
NO_HISTORY_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedNoHistory-Play-v0"
WIDE_DELAY_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedWideDelay-v0"
WIDE_DELAY_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedWideDelay-Play-v0"
SEAM_DEGRADED_TRAIN_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedSeam-v0"
SEAM_DEGRADED_PLAY_TASK_ID = "Marine-UR3-Random6DoFBase-WorldLineDegradedSeam-Play-v0"
UVMS_TRAIN_TASK_ID = "Marine-UR3-Uvms-WorldLineFreeFloating-v0"
UVMS_PLAY_TASK_ID = "Marine-UR3-Uvms-WorldLineFreeFloating-Play-v0"
UVMS_DRAG_TRAIN_TASK_ID = "Marine-UR3-Uvms-WorldLineFreeFloatingDragRandom-v0"
UVMS_DRAG_PLAY_TASK_ID = "Marine-UR3-Uvms-WorldLineFreeFloatingDragRandom-Play-v0"

def _register(
    task_id: str,
    env_cfg_name: str,
    entry_point: str = "isaaclab.envs:ManagerBasedRLEnv",
    runner_cfg_name: str = "RandomBaseWorldLinePPORunnerCfg",
) -> None:
    gym.register(
        id=task_id,
        entry_point=entry_point,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg_name}",
            "rsl_rl_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_cfg_name}"
            ),
        },
    )


_register(TRAIN_TASK_ID, "RandomBaseWorldLineEnvCfg")
_register(PLAY_TASK_ID, "RandomBaseWorldLineEnvCfg_PLAY")
_register(
    PRECISION_TRAIN_TASK_ID,
    "PrecisionStartWorldLineEnvCfg",
    f"{__name__}.env:PrecisionStartRLEnv",
)
_register(
    PRECISION_PLAY_TASK_ID,
    "PrecisionStartWorldLineEnvCfg_PLAY",
    f"{__name__}.env:PrecisionStartRLEnv",
)
_register(
    TARGET_CONDITIONED_TRAIN_TASK_ID,
    "TargetConditionedWorldLineEnvCfg",
    f"{__name__}.env:TargetConditionedRLEnv",
)
_register(
    TARGET_CONDITIONED_PLAY_TASK_ID,
    "TargetConditionedWorldLineEnvCfg_PLAY",
    f"{__name__}.env:TargetConditionedRLEnv",
)
_register(
    PRECISION_FAR_TRAIN_TASK_ID,
     "PrecisionFarImmediateBaseEnvCfg",
    f"{__name__}.env:PrecisionFarImmediateBaseRLEnv",
)
_register(
    PRECISION_FAR_PLAY_TASK_ID,
     "PrecisionFarImmediateBaseEnvCfg_PLAY",
    f"{__name__}.env:PrecisionFarImmediateBaseRLEnv",
)
# Success-biased continuation experiment: reuse the target-conditioned env class
# (rate limiter + immediate base motion) but with boosted tracking/success rewards.
_register(
    SUCCESS_BIAS_TRAIN_TASK_ID,
     "TargetCondWorldLineSuccessBiasEnvCfg",
    f"{__name__}.env:TargetConditionedRLEnv",
)
_register(
    SUCCESS_BIAS_PLAY_TASK_ID,
     "TargetCondWorldLineSuccessBiasEnvCfg_PLAY",
    f"{__name__}.env:TargetConditionedRLEnv",
)

# IK-seeded start + tool-axis constraint + raw-action penalties.
_register(
    IK_SEEDED_TRAIN_TASK_ID,
    "IkSeededWorldLineEnvCfg",
    f"{__name__}.env:IkSeededLineRLEnv",
    runner_cfg_name="IkSeededWorldLinePPORunnerCfg",
)
_register(
    IK_SEEDED_PLAY_TASK_ID,
    "IkSeededWorldLineEnvCfg_PLAY",
    f"{__name__}.env:IkSeededLineRLEnv",
    runner_cfg_name="IkSeededWorldLinePPORunnerCfg",
)

# Delayed, noisy base measurement shared by the policy and the model-based baseline.
_register(
    SENSOR_DEGRADED_TRAIN_TASK_ID,
    "SensorDegradedWorldLineEnvCfg",
    f"{__name__}.env:SensorDegradedLineRLEnv",
    runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
)
_register(
    SENSOR_DEGRADED_PLAY_TASK_ID,
    "SensorDegradedWorldLineEnvCfg_PLAY",
    f"{__name__}.env:SensorDegradedLineRLEnv",
    runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
)

# Ablations held ready: one isolates the measurement history, one widens the delay range.
for _train_id, _play_id, _cfg in (
    (NO_HISTORY_TRAIN_TASK_ID, NO_HISTORY_PLAY_TASK_ID, "SensorDegradedNoHistoryEnvCfg"),
    (WIDE_DELAY_TRAIN_TASK_ID, WIDE_DELAY_PLAY_TASK_ID, "SensorDegradedWideDelayEnvCfg"),
):
    _register(
        _train_id,
        _cfg,
        f"{__name__}.env:SensorDegradedLineRLEnv",
        runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
    )
    _register(
        _play_id,
        f"{_cfg}_PLAY",
        f"{__name__}.env:SensorDegradedLineRLEnv",
        runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
    )

# Both measurement channels degraded: base motion and the seam offset.
_register(
    SEAM_DEGRADED_TRAIN_TASK_ID,
    "SensorDegradedSeamEnvCfg",
    f"{__name__}.env:SensorDegradedLineRLEnv",
    runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
)
_register(
    SEAM_DEGRADED_PLAY_TASK_ID,
    "SensorDegradedSeamEnvCfg_PLAY",
    f"{__name__}.env:SensorDegradedLineRLEnv",
    runner_cfg_name="SensorDegradedWorldLinePPORunnerCfg",
)


# Free-floating vehicle. The arm's own motion pushes the hull, so reaching the line
# requires knowing what the next command does to the base one is standing on.
for _train_id, _play_id, _cfg in (
    (UVMS_TRAIN_TASK_ID, UVMS_PLAY_TASK_ID, "UvmsWorldLineEnvCfg"),
    (UVMS_DRAG_TRAIN_TASK_ID, UVMS_DRAG_PLAY_TASK_ID, "UvmsDragRandomizedEnvCfg"),
):
    _register(
        _train_id,
        _cfg,
        f"{__name__}.env:UvmsLineRLEnv",
        runner_cfg_name="UvmsWorldLinePPORunnerCfg",
    )
    _register(
        _play_id,
        f"{_cfg}_PLAY",
        f"{__name__}.env:UvmsLineRLEnv",
        runner_cfg_name="UvmsWorldLinePPORunnerCfg",
    )


# Stage 0's mass sweep: the same free-floating task carrying a lighter or heavier hull.
for _mass in (50, 200, 500, 2000):
    _register(
        f"Marine-UR3-Uvms-WorldLineFreeFloating{_mass}kg-Play-v0",
        f"UvmsWorldLine{_mass}kgEnvCfg_PLAY",
        f"{__name__}.env:UvmsLineRLEnv",
        runner_cfg_name="UvmsWorldLinePPORunnerCfg",
    )


# Residual policy on the analytic controller, under actuation delay: the condition
# finding 17 identified as the one where a policy has headroom to recover.
RESIDUAL_DELAY_TRAIN_TASK_ID = "Marine-UR3-ResidualIkDelay-v0"
RESIDUAL_DELAY_PLAY_TASK_ID = "Marine-UR3-ResidualIkDelay-Play-v0"
_register(
    RESIDUAL_DELAY_TRAIN_TASK_ID,
    "ResidualIkDelayEnvCfg",
    f"{__name__}.env:ResidualIkLineRLEnv",
    runner_cfg_name="ResidualIkRecurrentPPORunnerCfg",
)
_register(
    RESIDUAL_DELAY_PLAY_TASK_ID,
    "ResidualIkDelayEnvCfg_PLAY",
    f"{__name__}.env:ResidualIkLineRLEnv",
    runner_cfg_name="ResidualIkRecurrentPPORunnerCfg",
)
