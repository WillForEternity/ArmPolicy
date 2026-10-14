"""Verify calibration poses P1/P2/P3 and joint sign conventions by forward kinematics in the sim."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app
import math, numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply
from unoq_task import JOINTS, UnoQPickEnvCfg
D = math.radians
poses = {
    "P1_stretched_fwd": [0, 74.1, -75.5, 0, 0, -10],
    "P2_upper_vert_fore_horiz": [0, -15.9, 14.5, 0, 0, -10],
    "P3": [-35, -15.9, 14.5, 40, 45, -10],
    "roll+45_only": [0, -15.9, 14.5, 0, 45, -10],
    "grip_open100": [0, -15.9, 14.5, 0, 0, 100],
}
cfg = UnoQPickEnvCfg(); cfg.scene.num_envs = len(poses); cfg.scene.overhead = None
cfg.scene.robot.spawn.rigid_props.disable_gravity = True
cfg.scene.box.init_state.pos = (0.0, 0.5, 0.0127)
env = ManagerBasedRLEnv(cfg); env.reset()
cfg.events.reset_box = None
robot = env.scene["robot"]; names = robot.data.body_names; jids = [robot.data.joint_names.index(j) for j in JOINTS]
q = torch.tensor([[D(v) for v in p] for p in poses.values()], device=env.device)
for _ in range(90):
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=jids); env.step(q)
o = env.scene.env_origins
P = {n: (robot.data.body_pos_w[:, i] - o).cpu().numpy() for i, n in enumerate(names)}
gq = robot.data.body_quat_w[:, names.index("gripper")]
fing = quat_apply(gq, torch.tensor([0, 0, -1.0], device=env.device).expand(len(poses), 3)).cpu().numpy()
close_ax = quat_apply(gq, torch.tensor([1.0, 0, 0], device=env.device).expand(len(poses), 3)).cpu().numpy()
def ang(v):  # elevation (deg, + = up) and azimuth from forward(-y) toward robot-left(+x)
    return round(math.degrees(math.atan2(v[2], math.hypot(v[0], v[1]))), 1), round(math.degrees(math.atan2(v[0], -v[1])), 1)
print("pan axis xy (shoulder origin)", P["shoulder"][0][:2].round(4))
for i, k in enumerate(poses):
    ua = P["lower_arm"][i] - P["upper_arm"][i]; fa = P["wrist"][i] - P["lower_arm"][i]
    print(f"{k:26s} q_deg={poses[k]}  upper_arm(elev,az)={ang(ua)} forearm={ang(fa)} fingers={ang(fing[i])} "
          f"jaw_close_axis={np.round(close_ax[i],2)} moving_jaw_rel_gripper={np.round(P['jaw'][i]-P['gripper'][i],3)} "
          f"q_meas={np.round(np.degrees(robot.data.joint_pos[i, jids].cpu().numpy()),1)}")
print("FK_OK"); env.close(); app.close()
