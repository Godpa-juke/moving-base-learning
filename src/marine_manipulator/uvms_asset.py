"""The UR3 with its base released, carrying a vehicle's mass.

``ur3.usd`` anchors the arm: it carries a ``root_joint`` fixed to the world, and —
because of a PhysX parser limitation — the ``UsdPhysics.ArticulationRootAPI`` sits on
that joint rather than on a link. Setting
``ArticulationRootPropertiesCfg(fix_root_link=False)`` disables the joint but leaves
the articulation root on it, and PhysX then finds no articulation at all::

    Pattern '/World/envs/env_*/Robot/root_joint' did not match any rigid bodies
    RuntimeError: Failed to create articulation at: .../Robot/root_joint

Freeing the base therefore means moving the articulation root onto ``base_link``. An
articulation rooted on a rigid body is a floating-base articulation, which is exactly
what this round needs.

The vehicle is lumped into ``base_link`` rather than authored as a separate hull in
USD. That is a deliberate shortcut, stated as a limitation in ``docs/UVMS_PLAN.md``:
it changes neither the mass ratio nor the reaction the arm produces.
"""

from __future__ import annotations

from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR3_CFG
from pxr import PhysxSchema, UsdPhysics

#: Name of the fixed joint in ``ur3.usd`` that anchors the arm to the world.
ROOT_JOINT_NAME = "root_joint"

#: Name of the link the articulation root is moved onto, and which carries the vehicle.
BASE_LINK_NAME = "base_link"


@clone
def spawn_free_floating_usd(
    prim_path: str,
    cfg: "FreeFloatingUsdFileCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
):
    """Spawn the USD, then release its root and give ``base_link`` the vehicle's mass.

    The order matters. The articulation root has to move *before* the articulation
    properties are applied, or they land on the joint that is about to be deactivated.
    """
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)

    stage = get_current_stage()
    joint_prim = stage.GetPrimAtPath(f"{prim_path}/{cfg.root_joint_name}")
    if not joint_prim.IsValid():
        raise RuntimeError(
            f"'{prim_path}' has no '{cfg.root_joint_name}' to release; this spawner is "
            "specific to an asset whose articulation root sits on a world fixed joint"
        )
    # Deactivating removes both the fixed joint and the articulation root that USD put
    # on it. Disabling the joint alone would leave the root on a prim PhysX no longer
    # parses as an articulation.
    joint_prim.SetActive(False)

    base_path = f"{prim_path}/{cfg.base_link_name}"
    base_prim = stage.GetPrimAtPath(base_path)
    if not base_prim.IsValid():
        raise RuntimeError(f"'{base_path}' does not exist; cannot root the articulation there")
    UsdPhysics.ArticulationRootAPI.Apply(base_prim)
    PhysxSchema.PhysxArticulationAPI.Apply(base_prim)
    if cfg.articulation_props is not None:
        sim_utils.modify_articulation_root_properties(base_path, cfg.articulation_props, stage)

    if cfg.vehicle_mass_kg is not None:
        _set_vehicle_mass(stage, base_prim, cfg)
    return prim


def _set_vehicle_mass(stage, base_prim, cfg: "FreeFloatingUsdFileCfg") -> None:
    """Replace ``base_link``'s mass, and its inertia along with it.

    Swapping the mass without the inertia would leave a two-tonne body rotating like
    an arm link, which is the one thing this probe must not get wrong: the reaction the
    arm produces is as much a torque as a force.
    """
    mass_api = UsdPhysics.MassAPI.Apply(base_prim)
    stock_mass = mass_api.GetMassAttr().Get()
    mass_api.GetMassAttr().Set(float(cfg.vehicle_mass_kg))

    if cfg.vehicle_inertia_kg_m2 is not None:
        inertia = cfg.vehicle_inertia_kg_m2
    else:
        # No inertia given: keep the link's shape and change its density, which is the
        # weakest assumption available without authoring a hull.
        if not stock_mass:
            raise RuntimeError(
                f"'{base_prim.GetPath()}' has no authored mass to scale its inertia from; "
                "pass vehicle_inertia_kg_m2 explicitly"
            )
        stock_inertia = mass_api.GetDiagonalInertiaAttr().Get()
        ratio = float(cfg.vehicle_mass_kg) / float(stock_mass)
        inertia = tuple(float(value) * ratio for value in stock_inertia)
    mass_api.GetDiagonalInertiaAttr().Set(tuple(float(value) for value in inertia))


@configclass
class FreeFloatingUsdFileCfg(sim_utils.UsdFileCfg):
    """A :class:`UsdFileCfg` that moves the articulation root off the world joint."""

    func: Callable = spawn_free_floating_usd

    root_joint_name: str = ROOT_JOINT_NAME
    base_link_name: str = BASE_LINK_NAME

    #: Vehicle mass lumped into ``base_link``, kg. ``None`` leaves the asset's own.
    vehicle_mass_kg: float | None = None

    #: Vehicle diagonal inertia, kg m^2. ``None`` scales the asset's with the mass.
    vehicle_inertia_kg_m2: tuple[float, float, float] | None = None


def free_floating_ur3_cfg(
    vehicle_mass_kg: float | None = None,
    vehicle_inertia_kg_m2: tuple[float, float, float] | None = None,
    stiffness: float = 2000.0,
    damping: float = 100.0,
    disable_gravity: bool = True,
) -> ArticulationCfg:
    """The UR3 with a free root and the vehicle lumped into ``base_link``.

    Gravity defaults to *off*, which reverses what ``docs/UVMS_PLAN.md`` asked for.
    The reasoning is in :mod:`marine_manipulator.hydrodynamics`: folding added mass
    into the rigid-body mass inflates it by ~70%, so PhysX gravity would pull on a
    mass that buoyancy cannot balance, and ``disable_gravity`` is a per-USD property
    that cannot spare the arm links, which a near neutrally buoyant subsea arm needs.
    Weight, buoyancy and the righting moment between them are applied explicitly
    instead, so the restoring behaviour the plan wanted is present and correct.

    Passing ``disable_gravity=False`` restores the plan's literal arrangement; the
    Stage 0 probe uses it to measure a falling arm's reaction.

    The PD gains default to the ``UR3_HIGH_PD_CFG`` values so that the free-floating
    task inherits the same actuation as the fixed-base one it is compared against.
    They are a parameter here rather than an inherited constant because stiff joints
    transmit correspondingly large reaction torques into the hull.
    """
    cfg = UR3_CFG.copy()
    source = cfg.spawn
    cfg.spawn = FreeFloatingUsdFileCfg(
        usd_path=source.usd_path,
        activate_contact_sensors=source.activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=disable_gravity,
            max_depenetration_velocity=5.0,
        ),
        vehicle_mass_kg=vehicle_mass_kg,
        vehicle_inertia_kg_m2=vehicle_inertia_kg_m2,
    )
    # Without this the articulation view searches the tree, and the search is what
    # broke on the stock asset in the first place.
    cfg.articulation_root_prim_path = f"/{BASE_LINK_NAME}"
    cfg.actuators["arm"].stiffness = stiffness
    cfg.actuators["arm"].damping = damping
    return cfg


def uvms_robot_cfg(params=None, **kwargs) -> ArticulationCfg:
    """The free-floating UR3 carrying a specific vehicle, added mass included.

    The mass and inertia written into the USD are the *effective* ones — dry plus
    added — because added mass is absorbed into the rigid body rather than applied as
    a force. Everything downstream that needs the dry mass (weight, buoyancy) reads it
    from ``params`` rather than from the articulation, so the two never get confused.
    """
    if params is None:
        params = default_vehicle_params()
    return free_floating_ur3_cfg(
        vehicle_mass_kg=params.effective_mass_kg,
        vehicle_inertia_kg_m2=params.effective_inertia_kg_m2,
        **kwargs,
    )


def scaled_vehicle_params(mass_kg: float):
    """The nominal vehicle resized to ``mass_kg``, as a box of the same proportions.

    Used by the Stage 0 mass sweep. Inertia scales with the mass (same shape, denser
    body) and the drag coefficients scale with the projected area, which for a box of
    fixed proportions goes as ``mass^(2/3)``. Scaling drag with the mass instead would
    make a light vehicle both easier to push *and* easier to stop, confounding the very
    comparison the sweep is for.
    """
    from marine_manipulator import calibration
    from marine_manipulator.hydrodynamics import VehicleParams

    ratio = mass_kg / calibration.VEHICLE_MASS_KG
    area_ratio = ratio ** (2.0 / 3.0)
    return VehicleParams(
        mass_kg=mass_kg,
        inertia_kg_m2=tuple(v * ratio for v in calibration.VEHICLE_INERTIA_KG_M2),
        added_mass_fraction=calibration.ADDED_MASS_FRACTION,
        added_inertia_fraction=calibration.ADDED_INERTIA_FRACTION,
        linear_drag=tuple(v * area_ratio for v in calibration.LINEAR_DRAG),
        quadratic_drag=tuple(v * area_ratio for v in calibration.QUADRATIC_DRAG),
        cob_offset_b=calibration.COB_OFFSET_B,
        net_buoyancy_fraction=calibration.NET_BUOYANCY_FRACTION,
    )


def default_vehicle_params():
    """The nominal vehicle: the constants in :mod:`marine_manipulator.calibration`."""
    from marine_manipulator import calibration
    from marine_manipulator.hydrodynamics import VehicleParams

    return VehicleParams(
        mass_kg=calibration.VEHICLE_MASS_KG,
        inertia_kg_m2=calibration.VEHICLE_INERTIA_KG_M2,
        added_mass_fraction=calibration.ADDED_MASS_FRACTION,
        added_inertia_fraction=calibration.ADDED_INERTIA_FRACTION,
        linear_drag=calibration.LINEAR_DRAG,
        quadratic_drag=calibration.QUADRATIC_DRAG,
        cob_offset_b=calibration.COB_OFFSET_B,
        net_buoyancy_fraction=calibration.NET_BUOYANCY_FRACTION,
    )
