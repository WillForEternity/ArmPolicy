"""Generate scripted SO-101 UNO-Q-box pick demos in parallel (physics only, headless).

Planner: each episode samples its own grasp (which box axis to close across, which side the fixed jaw takes,
fixed-jaw clearance, off-centre offset, grasp height, gripper opening, approach tilt, hover/lift heights, speed).
Per control step (30 Hz) a TCP waypoint is min-jerk interpolated through phases; differential IK (damped least
squares on the gripper body Jacobian, integral action) turns it into absolute joint targets. DART-style noise is
added to the *executed* target while the *clean* planner target is recorded as the action. Only the physics
decides which grasps work; the sampled parameters are saved so the success of each can be analysed.

Output (per chunk, .npz): obs_joint_pos [N,T,6] rad (measured), obs_box [N,T,4] (x,y,z,yaw; pan-axis frame:
origin on the pan axis at table height, z up, forward=-y, +x=robot left, yaw CCW about +z from +x of the box long
axis), action [N,T,6] rad (clean commanded joint targets), length [N], success [N], gp_<name> [N] grasp params.
Joint order = JOINTS, angles in the SO-101 USD convention (convert with sim_bridge/).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=4)
parser.add_argument("--out", type=str, default="/root/isaac/data/sim_demos")
parser.add_argument("--noise_deg", type=float, default=1.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--grip_effort", type=float, default=0.30, help="gripper torque cap (N*m); 0.30 ~ 3.5 N squeeze")
parser.add_argument("--r_max", type=float, default=None)
parser.add_argument("--fixed_recipe", action="store_true", help="v1 behaviour: one grasp recipe for every episode")
parser.add_argument("--p_length_axis", type=float, default=0.0,
                    help="fraction of grasps closing across the 89 mm side (grasp search: 0/2057 succeed, jaws too small)")
parser.add_argument("--search", action="store_true", help="wide grasp-search ranges instead of the validated envelope")
parser.add_argument("--quiet", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)

from unoq_task import BOX_SIZE, JOINTS, PAN_AXIS_XY, REST_DEG, UnoQPickEnvCfg  # noqa: E402

torch.manual_seed(args.seed)
cfg = UnoQPickEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.scene.overhead = None
cfg.seed = args.seed
cfg.scene.robot.actuators["sts3215-gripper"].effort_limit_sim = args.grip_effort
if args.r_max is not None:
    cfg.events.reset_box.params["r_range"] = (0.12, args.r_max)
cfg.episode_length_s = 1e6  # episode boundaries are driven by the planner
env = ManagerBasedRLEnv(cfg)
dev, N = env.device, args.num_envs
robot, box = env.scene["robot"], env.scene["box"]
jids = [robot.data.joint_names.index(j) for j in JOINTS]
g_idx = robot.data.body_names.index("gripper")
jac_idx = g_idx - 1  # fixed base: PhysX jacobian rows skip the root body
lo, hi = robot.data.soft_joint_pos_limits[0, jids, 0], robot.data.soft_joint_pos_limits[0, jids, 1]

# Gripper geometry (probe_gripper.py): fixed-jaw tip at local x=-0.0088, z=-0.104; jaws close along +x.
FIXED_JAW_X, TCP_Z_LOCAL = -0.0088, -0.098
GRIP_CLOSE = math.radians(15)  # drive past contact; squeeze force is set by the effort cap, not the angle
# hand back (joint-space heuristic, runs identically on the real arm): from the lifted pose rotate shoulder_lift
# 45deg back/up, gripper stays closed.
HANDOFF_DQ = torch.tensor([0.0, math.radians(-45), 0.0, 0.0, 0.0, 0.0], device=dev)
PHASES = [("approach", 1.6), ("descend", 1.0), ("close", 1.2), ("lift", 1.0), ("handoff", 1.5), ("hold", 0.7)]
DT = env.step_dt
STEPS = [int(round(d / DT)) for _, d in PHASES]

U = lambda a, b: torch.empty(N, device=dev).uniform_(a, b)  # noqa: E731
tcp_local = torch.zeros(N, 3, device=dev)


def sample_grasp():
    """Per-episode grasp parameters. --fixed_recipe reproduces v1 (top-down, width axis, fixed numbers)."""
    if args.fixed_recipe:
        c = lambda v: torch.full((N,), float(v), device=dev)  # noqa: E731
        return dict(length_axis=c(0), flip=c(0), clearance=c(0.006), offset=c(0), grasp_z=c(0.0065),
                    open_deg=c(55), tilt_deg=c(0), hover_z=c(0.11), lift_z=c(0.12), speed=c(1.0))
    length_axis = (torch.rand(N, device=dev) < args.p_length_axis).float()
    if args.search:  # wide ranges used for the grasp search (results in ../README.md)
        return dict(length_axis=length_axis, flip=(torch.rand(N, device=dev) < 0.3).float(),
                    clearance=U(0.003, 0.012), offset=U(-0.3, 0.3), grasp_z=U(0.004, 0.011),
                    open_deg=torch.where(length_axis > 0, U(75, 88), U(50, 65)), tilt_deg=U(-15, 15),
                    hover_z=U(0.08, 0.13), lift_z=U(0.08, 0.14), speed=U(0.7, 1.0))
    # validated envelope from the grasp search (8192 grasps): everything here succeeds >=70% without noise
    return dict(
        length_axis=length_axis,                               # 0: close across the 66 mm side, 1: the 89 mm side
        flip=(torch.rand(N, device=dev) < 0.3).float(),        # 1: take the 180deg-other wrist yaw
        clearance=U(0.004, 0.010),                             # fixed jaw to box face before closing (m)
        offset=U(-0.3, 0.3),                                   # off-centre along the other box axis (fraction of half)
        grasp_z=U(0.004, 0.009),                               # TCP height at grasp (m)
        open_deg=torch.where(length_axis > 0, U(75, 88), U(52, 63)),
        tilt_deg=U(-4, 4),                                     # approach tilt about the jaw-closing axis
        hover_z=U(0.08, 0.13), lift_z=U(0.08, 0.14),
        speed=U(0.7, 1.0),                                     # phase duration fraction (holds after finishing)
    )


def tcp_pose():
    pos = robot.data.body_pos_w[:, g_idx] - env.scene.env_origins
    quat = robot.data.body_quat_w[:, g_idx]
    return pos + quat_apply(quat, tcp_local), quat


def closest_yaw(base, cur):
    cands = torch.stack([base, base + math.pi, base - math.pi, base + 2 * math.pi, base - 2 * math.pi], -1)
    diff = torch.atan2(torch.sin(cands - cur.unsqueeze(-1)), torch.cos(cands - cur.unsqueeze(-1))).abs()
    return cands.gather(1, diff.argmin(-1, keepdim=True)).squeeze(-1)


def min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def ik_step(target_tcp, target_quat, q):
    """One DLS IK update of the 5 arm joints toward the TCP pose, applied on top of the previous command q
    (integral action: cancels steady-state gravity sag of the position servos). Returns joint targets (6)."""
    jac = robot.root_physx_view.get_jacobians()[:, jac_idx, :, :][:, :, jids[:5]]
    body_pos = robot.data.body_pos_w[:, g_idx] - env.scene.env_origins
    body_quat = robot.data.body_quat_w[:, g_idx]
    target_body = target_tcp - quat_apply(target_quat, tcp_local)
    pe, re = compute_pose_error(body_pos, body_quat, target_body, target_quat, rot_error_type="axis_angle")
    err = torch.cat([pe, 0.5 * re], -1)  # 5-DOF arm: soften orientation so position wins
    W = torch.diag(torch.tensor([1, 1, 1, 0.5, 0.5, 0.5], device=dev))
    jt = jac.transpose(1, 2)
    dq = jt @ torch.linalg.solve(jac @ jt + 0.05**2 * torch.eye(6, device=dev), (W @ err.unsqueeze(-1)))
    out = q.clone()
    out[:, :5] = q[:, :5] + dq.squeeze(-1).clamp(-0.15, 0.15)
    return out


def _slerp_batch(q0, q1, s):
    """Per-env slerp; s is [N, 1]."""
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    th = torch.acos(dot.abs().clamp(max=1 - 1e-6))
    out = (torch.sin((1 - s) * th) * q0 + torch.sin(s * th) * q1) / torch.sin(th)
    return out / out.norm(dim=-1, keepdim=True)


def run_round(r):
    env.reset()
    # start near rest with per-episode jitter, like a human starting a teleop take
    q0 = torch.tensor([math.radians(REST_DEG[j]) for j in JOINTS], device=dev).repeat(N, 1)
    q0 += torch.randn_like(q0) * math.radians(4)
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=jids)
    for _ in range(15):
        env.step(q0)

    gp = sample_grasp()
    la = gp["length_axis"]
    half_close = torch.where(la > 0, torch.full_like(la, BOX_SIZE[0] / 2), torch.full_like(la, BOX_SIZE[1] / 2))
    half_other = torch.where(la > 0, torch.full_like(la, BOX_SIZE[1] / 2), torch.full_like(la, BOX_SIZE[0] / 2))
    tcp_local[:] = torch.stack([FIXED_JAW_X + half_close + gp["clearance"], gp["offset"] * half_other,
                                torch.full_like(la, TCP_Z_LOCAL)], -1)

    box_pos = box.data.root_pos_w - env.scene.env_origins
    _, _, box_yaw = euler_xyz_from_quat(box.data.root_quat_w)
    tcp0, quat0 = tcp_pose()
    _, _, cur_yaw = euler_xyz_from_quat(quat0)
    # closing axis (gripper local x) along the box short axis (yaw+90deg) or long axis (yaw)
    yaw_t = closest_yaw(box_yaw + (1 - la) * math.pi / 2, cur_yaw) + gp["flip"] * math.pi
    zero = torch.zeros_like(yaw_t)
    quat_t = quat_mul(quat_from_euler_xyz(zero, zero, yaw_t), quat_from_euler_xyz(torch.deg2rad(gp["tilt_deg"]), zero, zero))
    xy = box_pos[:, :2] + 0.004 * torch.randn(N, 2, device=dev)  # small grasp-point noise
    col = lambda v: v.unsqueeze(-1)  # noqa: E731
    hover = torch.cat([xy, col(gp["hover_z"])], -1)
    grasp = torch.cat([xy, col(gp["grasp_z"])], -1)
    lift = torch.cat([xy, col(gp["lift_z"])], -1)
    g_open = col(torch.deg2rad(gp["open_deg"]))
    g_close = torch.full((N, 1), GRIP_CLOSE, device=dev)

    noise_on = (torch.rand(N, 1, device=dev) < 0.7).float()  # 70% of episodes get DART noise
    obs_q, obs_b, acts = [], [], []
    q_cmd = robot.data.joint_pos[:, jids].clone()
    for name, n in zip([p for p, _ in PHASES], STEPS):
        start_tcp, _ = tcp_pose()
        q_start = q_cmd.clone()
        for k in range(n):
            s = min_jerk(((k + 1) / (n * col(gp["speed"]))).clamp(max=1.0))  # [N,1], faster episodes then hold
            if name == "approach":
                tgt, tq, grip = start_tcp + (hover - start_tcp) * s, _slerp_batch(quat0, quat_t, s), q0[:, 5:6] + (g_open - q0[:, 5:6]) * s
            elif name == "descend":
                tgt, tq, grip = hover + (grasp - hover) * s, quat_t, g_open
            elif name == "close":
                tgt, tq, grip = grasp, quat_t, g_open + (g_close - g_open) * s
            elif name == "lift":
                tgt, tq, grip = grasp + (lift - grasp) * s, quat_t, g_close
            else:  # handoff / hold: joint-space heuristic, no IK
                tgt, tq, grip = None, None, g_close
            q_now = robot.data.joint_pos[:, jids]
            if tgt is not None:
                q_cmd = ik_step(tgt, tq, q_cmd)
                q_cmd[:, :5] = torch.max(torch.min(q_cmd[:, :5], q_now[:, :5] + 0.35), q_now[:, :5] - 0.35)  # anti-windup
            elif name == "handoff":
                q_cmd = q_start + HANDOFF_DQ * s
            q_cmd[:, 5:6] = grip
            q_cmd = torch.max(torch.min(q_cmd, hi), lo)
            # record observation (pre-step) and clean action
            bp = box.data.root_pos_w - env.scene.env_origins - torch.tensor([PAN_AXIS_XY[0], PAN_AXIS_XY[1], 0.0], device=dev)
            _, _, by = euler_xyz_from_quat(box.data.root_quat_w)
            obs_q.append(q_now.clone())
            obs_b.append(torch.cat([bp, by.unsqueeze(-1)], -1))
            acts.append(q_cmd.clone())
            q_exec = q_cmd + noise_on * torch.randn_like(q_cmd) * math.radians(args.noise_deg) * torch.tensor(
                [1, 1, 1, 1, 1, 0], device=dev)
            env.step(q_exec)
        if not args.quiet:
            tcp_e, _ = tcp_pose()
            pe = torch.linalg.norm(tcp_e - tgt, dim=-1).mean() if tgt is not None else torch.tensor(0.0)
            bzz = (box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean()
            print(f"  [{name}] tcp_err {pe:.4f} | grip {math.degrees(robot.data.joint_pos[:, jids[5]].mean()):.1f} "
                  f"| box_z {bzz:.4f}", flush=True)

    bz = box.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    tcp, _ = tcp_pose()
    near = torch.linalg.norm((box.data.root_pos_w - env.scene.env_origins) - tcp, dim=-1) < 0.07
    success = (bz > BOX_SIZE[2] / 2 + 0.10) & near  # still held at the hand-off pose
    gp_np = {k: v.cpu().numpy() for k, v in gp.items()}
    gp_np["r"] = np.hypot(obs_b[0][:, 0].cpu().numpy(), obs_b[0][:, 1].cpu().numpy())
    return (torch.stack(obs_q, 1).cpu().numpy(), torch.stack(obs_b, 1).cpu().numpy(),
            torch.stack(acts, 1).cpu().numpy(), success.cpu().numpy(), gp_np)


def report(gps, ok):
    print("=== success by grasp parameter ===")
    for k, v in gps.items():
        if len(np.unique(v)) <= 2:
            print(f"  {k:12s} " + "  ".join(f"{u:g}: {ok[v == u].mean():.2f} (n={int((v == u).sum())})" for u in np.unique(v)))
        else:
            edges = np.quantile(v, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
            idx = np.clip(np.digitize(v, edges[1:-1]), 0, 4)
            print(f"  {k:12s} " + "  ".join(f"[{edges[i]:.3g},{edges[i+1]:.3g}]: {ok[idx == i].mean():.2f}" for i in range(5)))


os.makedirs(args.out, exist_ok=True)
tot_s = tot = 0
all_gp, all_ok = {}, []
for r in range(args.rounds):
    t0 = time.time()
    oq, ob, a, ok, gp = run_round(r)
    np.savez_compressed(f"{args.out}/chunk_s{args.seed}_r{r:03d}.npz", obs_joint_pos=oq, obs_box=ob, action=a,
                        success=ok, length=np.full(len(ok), oq.shape[1]), phase_steps=np.array(STEPS),
                        phase_names=np.array([n for n, _ in PHASES]), frame=np.array("pan_axis"),
                        joint_names=np.array(JOINTS), fps=np.array(1 / DT), box_size=np.array(BOX_SIZE),
                        **{f"gp_{k}": v for k, v in gp.items()})
    for k, v in gp.items():
        all_gp.setdefault(k, []).append(v)
    all_ok.append(ok)
    tot_s += ok.sum(); tot += len(ok)
    print(f"ROUND {r}: success {ok.mean():.3f} ({ok.sum()}/{len(ok)}) T={oq.shape[1]} in {time.time()-t0:.1f}s "
          f"cum {tot_s}/{tot}", flush=True)
report({k: np.concatenate(v) for k, v in all_gp.items()}, np.concatenate(all_ok))
print("GEN_DONE", tot_s, tot)
env.close()
app.close()
