"""Boot Isaac Sim headless, build the UNO Q scene, dump SO-101 kinematics, save an overhead render."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import math  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from PIL import Image  # noqa: E402

from unoq_task import JOINTS, UnoQPickEnvCfg  # noqa: E402

cfg = UnoQPickEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg)
obs, _ = env.reset()
robot = env.scene["robot"]
print("JOINT NAMES", robot.data.joint_names)
print("BODY NAMES", robot.data.body_names)
print("JOINT LIMITS deg", (torch.rad2deg(robot.data.joint_pos_limits[0])).round().tolist())
ids = [robot.data.joint_names.index(j) for j in JOINTS]
origin = env.scene.env_origins[0]


def settle(q, steps=60):
    act = torch.tensor(q, device=env.device).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(act)


def bodies():
    return {n: (robot.data.body_pos_w[0, i] - origin).cpu().numpy().round(4).tolist()
            for i, n in enumerate(robot.data.body_names)}


for name, q in {
    "zero": [0, 0, 0, 0, 0, 0],
    "rest": [math.radians(v) for v in [0, -100, 90, 50, 0, -10]],
    "pan+45": [math.radians(45), 0, 0, 0, 0, 0],
    "reach_down": [0, math.radians(30), math.radians(-10), math.radians(70), 0, math.radians(60)],
}.items():
    settle(q)
    print("POSE", name, "joint_pos_deg", torch.rad2deg(robot.data.joint_pos[0, ids]).round().tolist())
    print("  BODIES", bodies())
    g = robot.data.body_names.index("gripper")
    print("  GRIPPER QUAT wxyz", robot.data.body_quat_w[0, g].cpu().numpy().round(3).tolist())

print("BOX POSE obs (x,y,z,cos2yaw,sin2yaw)", obs["policy"][:, 6:].cpu().numpy().round(3).tolist())
rgb = env.scene["overhead"].data.output["rgb"][0].cpu().numpy().astype(np.uint8)
Image.fromarray(rgb).save("/root/isaac/out/overhead.png")
print("RGB mean", rgb.mean(), rgb.shape)
print("PROBE_OK")
env.close()
app.close()
