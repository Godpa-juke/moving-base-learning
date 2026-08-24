"""Dependency-free geometric constants shared by the tasks, kinematics and tests.

These live outside :mod:`marine_manipulator.tasks.random_base_line.mdp` so that
kinematics tests can run without importing Isaac Lab (and therefore ``pxr``).
The TCP poses were measured inside the running simulator and are the reference
that :mod:`marine_manipulator.ur3_kin` is validated against.
"""

from __future__ import annotations

#: Cylinder tip offset along the flange (+Z of ``wrist_3_link``), metres.
TCP_OFFSET = (0.0, 0.0, 0.12)

PRECISION_START_HOLD_S = 1.0

#: Per-joint target speed cap used by the action rate limiter, rad/s.
REALISTIC_JOINT_SPEED_LIMIT_RAD_S = (0.6, 0.6, 0.6, 0.8, 0.8, 0.8)
REALISTIC_ACTION_SCALE_RAD = 0.5

# Calibrated from the actual ManagerBased runtime with the verified start q.
# The line center is +0.10 m in Y from this -Y endpoint.
PRECISION_START_TCP_E = (0.30049800872802734, -0.12791526317596436, 0.2688601613044739)
PRECISION_FAR_START_TCP_E = (
    PRECISION_START_TCP_E[0] - 0.50,
    PRECISION_START_TCP_E[1],
    PRECISION_START_TCP_E[2],
)
PRECISION_START_JOINT_POS = (
    -1.0903867483139038,
    -1.3245214223861694,
    0.9494641423225403,
    -0.7210094332695007,
    -1.0808051824569702,
    0.0,
)
PRECISION_FAR_START_JOINT_POS = (
    0.5848665237426758,
    -3.0241305828094482,
    0.40810462832450867,
    0.030411597341299057,
    -2.1389567852020264,
    0.0,
)

#: Joint configuration that damped-least-squares IK can start from and still reach
#: every vertical-tool pose in the target workspace box. Found by searching joint
#: space with :func:`marine_manipulator.ur3_kin.sample_seed_bank`; a nominal
#: posture is unreliable: vertical-tool solutions live in a distant joint-space
#: branch, and seeding from a natural pose converges only intermittently
#: (see ``tests/test_ur3_kin.py``).
VERTICAL_TOOL_IK_SEED = (
    -1.2899,
    0.3815,
    1.6528,
    1.6296,
    -2.7456,
    -2.78,
)

#: Workspace the target line actually sweeps, in the robot root frame (metres).
#: The command samples a line centre in x (-0.40, -0.30), y (-0.03, 0.03),
#: z (0.18, 0.24) and then traverses +-amplitude (0.05..0.07) along y, so y spans
#: (-0.10, 0.10) over an episode.
TARGET_BOX_X = (-0.40, -0.30)
TARGET_BOX_Y = (-0.10, 0.10)
TARGET_BOX_Z = (0.18, 0.24)

#: Same workspace pulled 4 cm toward the base. The far edge of TARGET_BOX_* sits
#: 0.455 m from the shoulder against a two-link reach of |A2| + |A3| = 0.457 m, so
#: the baseline task is trained a millimetre inside the kinematic limit, where the
#: Jacobian's smallest singular value falls to 0.070. Pulling in lifts the
#: worst case to 0.088 (+24%) at the cost of a slightly smaller workspace.
INBOARD_BOX_X = (-0.36, -0.28)
INBOARD_BOX_Y = (-0.09, 0.09)
INBOARD_BOX_Z = (0.19, 0.25)

#: Joint configuration whose cylinder tip sits at the centre of TARGET_BOX_* with
#: the tool vertical. Used as the articulation default so that the joint-position
#: action space is centred on the task instead of on an unrelated posture.
BOX_CENTER_JOINT_POS = (
    2.8148074334,
    -0.1935551115,
    -1.3442812031,
    -0.0329600180,
    -1.5707963242,
    1.2440111061,
)

# --------------------------------------------------------------------------------------
# Sensor degradation.
#
# Baseline observations hand the policy the exact base motion, analytically evaluated
# from the sampled amplitude/frequency/phase, and the IK baseline reads the true root
# pose. Neither exists on a vessel: an INS or a seam tracker reports late and noisy.
# Randomising over these ranges lets one policy be evaluated across the whole sweep,
# and the same measurement feeds the model-based controller so the comparison is fair.
# --------------------------------------------------------------------------------------

#: Measurement delay, seconds. Upper bound is 4 control steps at 30 Hz.
SENSOR_DELAY_RANGE_S = (0.0, 0.1333)

#: Additive Gaussian noise on the measured base translation, metres (std).
SENSOR_POSITION_NOISE_RANGE_M = (0.0, 0.002)

#: Additive Gaussian noise on the measured base rotation, radians (std).
SENSOR_ROTATION_NOISE_RANGE_RAD = (0.0, 0.003)

#: Frames of measured base state the policy sees. At 30 Hz, 20 frames span 0.67 s,
#: enough of a 3-12 s disturbance period to estimate its phase and frequency and so
#: predict through the delay. A handful of consecutive frames would not be.
SENSOR_HISTORY_LENGTH = 20

#: Ring-buffer depth for the seam measurement, in control steps. Must cover the
#: largest delay any task samples (0.3333 s at 30 Hz = 10 steps) plus the current frame.
SEAM_BUFFER_STEPS = 12


# --------------------------------------------------------------------------------------
# Free-floating vehicle (UVMS round).
#
# The arm is released from the world and lumped together with a vehicle hull inside
# `base_link`. Stage 0 measured the coupling as a function of the vehicle's mass; the
# value chosen here is the smallest of the credible classes, because the coupling this
# round is about is strongest there and the plan's kill criterion is about whether it
# exists at all.
#
# Inertia is authored explicitly rather than scaled from the stock `base_link`. Scaling
# a 2 kg UR3 base plate up to 100 kg keeps a plate's rotational inertia, which is about
# twenty times too small for a hull of this size and makes the arm's reaction torque
# look far more effective than it is. The values below are a 0.7 x 0.5 x 0.4 m box of
# uniform density, which is the crudest defensible model of an observation-class ROV.
# --------------------------------------------------------------------------------------

#: Vehicle dry mass, kg. Observation-class ROV; see `docs/UVMS_PLAN.md` stage 0.
VEHICLE_MASS_KG = 100.0

#: Diagonal inertia of a 100 kg, 0.7 x 0.5 x 0.4 m uniform box, kg m^2.
VEHICLE_INERTIA_KG_M2 = (3.42, 5.42, 6.17)

#: Translational added mass as a fraction of the dry mass. A bluff submerged body sits
#: near 0.3 in surge and near 1.0 in heave; PhysX carries a single scalar mass per body,
#: so the anisotropy is replaced by its mean and that is recorded as an approximation.
ADDED_MASS_FRACTION = 0.7

#: Rotational added inertia as a fraction of the dry inertia, per body axis.
ADDED_INERTIA_FRACTION = (0.4, 0.5, 0.6)

#: Fossen linear drag diagonal: N s/m for surge/sway/heave, N m s/rad for roll/pitch/yaw.
LINEAR_DRAG = (30.0, 40.0, 50.0, 5.0, 6.0, 7.0)

#: Fossen quadratic drag diagonal: N/(m/s)^2 and N m/(rad/s)^2. Translational entries
#: are 0.5 * rho * C_d * A for sea water at C_d ~ 1 and the box's projected areas.
QUADRATIC_DRAG = (180.0, 220.0, 260.0, 20.0, 25.0, 30.0)

#: Centre of buoyancy above the centre of gravity, body frame, metres. This offset is
#: the vehicle's only passive stability in roll and pitch; at neutral buoyancy it gives
#: a righting stiffness of m g h = 58.9 N m/rad.
COB_OFFSET_B = (0.0, 0.0, 0.06)

#: Buoyancy in excess of weight. Zero because the vehicle has no thrusters: a real ROV
#: runs 1-2% positive and trims it out with thrust, which this model has no way to do,
#: so any non-zero value integrates into an unbounded rise over an episode.
NET_BUOYANCY_FRACTION = 0.0

#: Multiplicative error applied to the drag coefficients the *simulation* uses, while
#: the controller and the policy keep the nominal values above. This is the Stage 3
#: x-axis. 1.0 is no mismatch; 0.5 and 2.0 are the +-100% ends.
DRAG_MISMATCH_TRAIN_RANGE = (0.5, 2.0)
