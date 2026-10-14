"""SO-101 + Arduino UNO Q box pick task (Isaac Lab manager-based env).

World frame == robot base frame: robot base at the origin on the table top (z = 0).
Action = absolute joint position targets for the 6 SO-101 joints (rad), exactly what a
leader arm / learned policy commands on the real follower. Control rate 30 Hz (matches LeRobot teleop).
"""

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from leisaac.assets.robots.lerobot import SO101_FOLLOWER_CFG

# Arduino UNO Q retail box, measured from the iPhone overhead frame (perspective-rectified lid, scale from the
# 1:1 printed UNO-footprint board): 88.9 x 65.8 mm lid, height 0.38 x width (+-3 mm)
BOX_SIZE = (0.089, 0.066, 0.025)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST_DEG = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 90.0, "wrist_flex": 50.0,
            "wrist_roll": 0.0, "gripper": -10.0}

# Data frame (shared with the real pipeline, see CONTRACT.md): origin on the pan axis at table height,
# z up, forward = -y, +x = robot's left. The pan axis sits at PAN_AXIS_XY in the SO-101 USD base frame.
PAN_AXIS_XY = (0.0208, -0.0231)
SPAWN_FORWARD = (0.0, -1.0)  # SO-101 USD reaches along -y of its base frame
SPAWN_R = (0.12, 0.28)      # m from the pan axis; top-down grasp success falls off past 0.28 m
SPAWN_BEARING = (-0.6, 0.6)  # rad left/right of forward


def box_pose_xy_yaw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("box")) -> torch.Tensor:
    """(x, y, z, cos 2yaw, sin 2yaw) of the box in the pan-axis frame. 2*yaw because the box is 180deg symmetric."""
    box: RigidObject = env.scene[asset_cfg.name]
    pos = box.data.root_pos_w - env.scene.env_origins
    pos[:, 0] -= PAN_AXIS_XY[0]
    pos[:, 1] -= PAN_AXIS_XY[1]
    _, _, yaw = euler_xyz_from_quat(box.data.root_quat_w)
    return torch.cat([pos, torch.cos(2 * yaw).unsqueeze(-1), torch.sin(2 * yaw).unsqueeze(-1)], dim=-1)


def reset_box_polar(env, env_ids, r_range, bearing_range, forward, asset_cfg=SceneEntityCfg("box")):
    box: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    dev = env.device
    r = torch.empty(n, device=dev).uniform_(*r_range)
    fwd = math.atan2(forward[1], forward[0])
    th = fwd + torch.empty(n, device=dev).uniform_(*bearing_range)
    yaw = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
    state = box.data.default_root_state[env_ids].clone()
    state[:, 0] = PAN_AXIS_XY[0] + r * torch.cos(th)
    state[:, 1] = PAN_AXIS_XY[1] + r * torch.sin(th)
    state[:, 2] = BOX_SIZE[2] / 2 + 0.001
    state[:, :3] += env.scene.env_origins[env_ids]
    state[:, 3] = torch.cos(yaw / 2)
    state[:, 4:6] = 0.0
    state[:, 6] = torch.sin(yaw / 2)
    state[:, 7:] = 0.0
    box.write_root_state_to_sim(state, env_ids=env_ids)


def box_lifted(env, height: float = 0.06, asset_cfg=SceneEntityCfg("box")) -> torch.Tensor:
    box: RigidObject = env.scene[asset_cfg.name]
    return (box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) > height


@configclass
class UnoQSceneCfg(InteractiveSceneCfg):
    robot = SO101_FOLLOWER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_FOLLOWER_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={k: math.radians(v) for k, v in REST_DEG.items()},
        ),
    )

    # dark tablecloth: static collider whose top face is z = 0
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.2, -0.02)),
        spawn=sim_utils.CuboidCfg(
            size=(0.9, 0.9, 0.04),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.035), roughness=0.9),
        ),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.22, BOX_SIZE[2] / 2 + 0.001)),
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.88), roughness=0.6),
        ),
    )

    overhead = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.22, 0.75), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 5.0)),
        width=640, height=480,
    )

    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)})
        box_pose = ObsTerm(func=box_pose_xy_yaw)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    joints = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINTS, preserve_order=True, scale=1.0,
                                        use_default_offset=False)


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_box = EventTerm(func=reset_box_polar, mode="reset",
                          params={"r_range": SPAWN_R, "bearing_range": SPAWN_BEARING, "forward": SPAWN_FORWARD})
    box_mass = EventTerm(func=mdp.randomize_rigid_body_mass, mode="startup",
                         params={"asset_cfg": SceneEntityCfg("box"), "mass_distribution_params": (0.05, 0.12),
                                 "operation": "abs"})
    box_friction = EventTerm(func=mdp.randomize_rigid_body_material, mode="startup",
                             params={"asset_cfg": SceneEntityCfg("box"), "static_friction_range": (0.35, 0.6),
                                     "dynamic_friction_range": (0.3, 0.5), "restitution_range": (0.0, 0.05),
                                     "num_buckets": 64})


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class UnoQPickEnvCfg(ManagerBasedRLEnvCfg):
    scene: UnoQSceneCfg = UnoQSceneCfg(num_envs=64, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        self.sim.dt = 1 / 120
        self.decimation = 4  # 30 Hz control, same as LeRobot teleop
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (0.5, -0.7, 0.5)
        self.viewer.lookat = (0.0, -0.18, 0.05)
        # Paper box: cap gripper torque at 0.30 N*m (~3.5 N squeeze at ~8.5 cm lever). Swept 0.2-1.0 N*m: higher caps
        # wedge the box out of the jaws, lower ones slip. Real follower: STS3215 torque limit ~15% gives the same cap.
        self.scene.robot.actuators["sts3215-gripper"].effort_limit_sim = 0.30
        self.scene.robot.actuators["sts3215-arm"].effort_limit_sim = 3.0
