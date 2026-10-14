"""Fingertip geometry of the SO-101 gripper: mesh points in gripper/jaw body frames, opening vs gripper angle."""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p); a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import math
import numpy as np
import torch
from pxr import Usd, UsdGeom
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_inv
import isaaclab.sim as sim_utils
from unoq_task import UnoQPickEnvCfg

cfg = UnoQPickEnvCfg(); cfg.scene.num_envs = 1; cfg.scene.overhead = None
env = ManagerBasedRLEnv(cfg); env.reset()
stage = sim_utils.get_current_stage()
robot = env.scene["robot"]
names = robot.data.body_names


def body_points(body):
    root = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body}")
    body_xf = UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    pts = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and "collision" not in str(prim.GetPath()).lower():
            m = UsdGeom.Mesh(prim)
            v = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * body_xf
            M = np.array(xf).T  # usd row-vector convention
            vh = np.c_[v, np.ones(len(v))] @ M.T
            pts.append(vh[:, :3])
    return np.concatenate(pts)


for q_grip in [0, 20, 40, 60, 80, 100]:
    q = torch.tensor([[0, math.radians(30), math.radians(-10), math.radians(70), 0, math.radians(q_grip)]], device=env.device)
    for _ in range(40):
        env.step(q)
    if q_grip == 0:
        G = body_points("gripper"); J = body_points("jaw")
        print("gripper local bbox min", G.min(0).round(4), "max", G.max(0).round(4), "n", len(G))
        print("jaw local bbox min", J.min(0).round(4), "max", J.max(0).round(4), "n", len(J))
    gi, ji = names.index("gripper"), names.index("jaw")
    gp, gq = robot.data.body_pos_w[0, gi], robot.data.body_quat_w[0, gi]
    jp, jq = robot.data.body_pos_w[0, ji], robot.data.body_quat_w[0, ji]
    Gw = (quat_apply(gq.repeat(len(G), 1), torch.tensor(G, dtype=torch.float32, device=env.device)) + gp).cpu().numpy()
    Jw = (quat_apply(jq.repeat(len(J), 1), torch.tensor(J, dtype=torch.float32, device=env.device)) + jp).cpu().numpy()
    gl, jl = Gw[Gw[:, 2].argsort()[:30]].mean(0), Jw[Jw[:, 2].argsort()[:30]].mean(0)
    print(f"grip={q_grip:3d}deg actual={math.degrees(robot.data.joint_pos[0,5]):6.1f} fixed_tip={gl.round(4)} jaw_tip={jl.round(4)} tip_dist={np.linalg.norm(gl-jl):.4f}")
    # tip points expressed in the gripper body frame
    to_local = lambda pw: quat_apply(quat_inv(gq).unsqueeze(0), torch.tensor(pw, dtype=torch.float32, device=env.device).unsqueeze(0) - gp).cpu().numpy().round(4)
    print("   in gripper frame: fixed_tip", to_local(gl), "jaw_tip", to_local(jl))
print("GRIP_PROBE_OK")
env.close(); app.close()
