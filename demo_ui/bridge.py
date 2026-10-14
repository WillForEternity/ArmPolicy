"""Telemetry bridge between the demo runtime and the browser UI. Standard library only.

The runtime pushes whatever it knows; the browser gets a merged snapshot over Server-Sent Events.
Nothing in here talks to servos or the board: commands from the UI (start, estop, loops, release)
are handed back to the runtime, which decides what to do with them.

    bus = DemoBus().start()                      # http://localhost:8765
    bus.publish(mode="live", phase="pick", state=q, action=a, chunk=c,
                box={"xy": [x, y], "theta_deg": th, "corners_px": quad, "latched": True},
                infer={"host": "uno-q", "ms": 3.1, "loops": 4, "model": "looped_k4_int8"},
                links={"camera": True, "follower": True, "board": True})
    bus.log("grasp detected")
    for cmd in bus.poll_commands(): ...          # {"cmd": "grab"} starts a pick, {"cmd": "estop"} etc.

Replay (no hardware; plays real Isaac Sim episodes and times the exported ONNX on this Mac):

    python -m demo_ui.bridge --replay [--on-demand]
"""

import argparse
import json
import mimetypes
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "web" / "dist"
SCORECARD = HERE / "scorecard.json"
PHASES = ("idle", "pick", "lift", "hold", "release", "home", "estop")


class DemoBus:
    def __init__(self, port=8765, hz=30.0):
        self.port, self.period = port, 1.0 / hz
        self.state: dict = {"mode": "live", "phase": "idle"}
        self.logs: deque = deque(maxlen=200)
        self._clients: list[queue.Queue] = []
        self._cmds: queue.Queue = queue.Queue()
        self._callbacks = []
        self._lock = threading.Lock()
        self._dirty = threading.Event()
        self._frame, self._frame_cv = None, threading.Condition()

    # ---- runtime side -------------------------------------------------------------------------
    def publish(self, **fields) -> None:
        with self._lock:
            for k, v in fields.items():
                if isinstance(v, dict) and isinstance(self.state.get(k), dict):
                    self.state[k] = {**self.state[k], **v}
                else:
                    self.state[k] = _plain(v)
            self.state["t"] = time.time()
        self._dirty.set()

    def log(self, text: str, level="info") -> None:
        entry = {"t": time.time(), "level": level, "text": text}
        with self._lock:
            self.logs.append(entry)
        self._send("log", entry)

    def publish_frame(self, jpeg: bytes) -> None:
        """Optional: annotated camera frames for /video.mjpg (if the browser can't open the camera)."""
        first = self._frame is None
        with self._frame_cv:
            self._frame = jpeg
            self._frame_cv.notify_all()
        if first:
            self.publish(video=True)  # tells the UI to show /video.mjpg instead of opening the camera itself

    def poll_commands(self) -> list[dict]:
        out = []
        while not self._cmds.empty():
            out.append(self._cmds.get_nowait())
        return out

    def on_command(self, fn) -> None:
        self._callbacks.append(fn)

    def start(self) -> "DemoBus":
        bus = self

        class Handler(_Handler):
            pass

        Handler.bus = bus
        self.server = _Server(("127.0.0.1", self.port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._pump, daemon=True).start()
        print(f"demo UI: http://localhost:{self.port}")
        return self

    # ---- internals ----------------------------------------------------------------------------
    def _pump(self) -> None:
        while True:
            self._dirty.wait()
            self._dirty.clear()
            with self._lock:
                snap = dict(self.state)
            self._send("state", snap)
            time.sleep(self.period)

    def _send(self, event: str, data) -> None:
        msg = f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode()
        for q in list(self._clients):
            if q.qsize() < 64:  # slow client: drop rather than grow
                q.put(msg)

    def _command(self, cmd: dict) -> None:
        self._cmds.put(cmd)
        self.log(f"ui → {cmd.get('cmd')}" + (f" {cmd['k']}" if "k" in cmd else ""), "cmd")
        for fn in self._callbacks:
            fn(cmd)


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        import sys

        if not isinstance(sys.exc_info()[1], (ConnectionError, BrokenPipeError)):  # browsers drop streams all the time
            super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    bus: DemoBus

    def log_message(self, *a):
        pass

    def _head(self, code=200, ctype="application/json", extra=()):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()

    def do_OPTIONS(self):
        self._head(204)

    def do_POST(self):
        if self.path != "/cmd":
            return self._head(404)
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        try:
            self.bus._command(json.loads(body or b"{}"))
        except json.JSONDecodeError:
            return self._head(400)
        self._head(204)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/events":
            return self._events()
        if path == "/video.mjpg":
            return self._mjpeg()
        if path == "/frame.jpg":  # latest frame; the UI re-requests on load, so it self-paces
            jpg = self.bus._frame
            if not jpg:
                return self._head(404)
            self._head(200, "image/jpeg", [("Cache-Control", "no-store"), ("Content-Length", str(len(jpg)))])
            self.wfile.write(jpg)
            return
        if path == "/scorecard.json":
            self._head()
            self.wfile.write(SCORECARD.read_bytes() if SCORECARD.exists() else b"{}")
            return
        self._static(path)

    def _events(self):
        self._head(200, "text/event-stream", [("Cache-Control", "no-cache"), ("Connection", "keep-alive")])
        q: queue.Queue = queue.Queue()
        bus = self.bus
        with bus._lock:
            first = [("state", dict(bus.state))] + [("log", e) for e in list(bus.logs)[-30:]]
        for ev, d in first:
            q.put(f"event: {ev}\ndata: {json.dumps(d)}\n\n".encode())
        bus._clients.append(q)
        try:
            while True:
                try:
                    self.wfile.write(q.get(timeout=5))
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            bus._clients.remove(q)

    def _mjpeg(self):
        self._head(200, "multipart/x-mixed-replace; boundary=frame", [("Cache-Control", "no-cache")])
        cv = self.bus._frame_cv
        try:
            while True:
                with cv:
                    cv.wait(timeout=2)
                    jpg = self.bus._frame
                if jpg:
                    head = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpg)}\r\n\r\n".encode()
                    self.wfile.write(head + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _static(self, path: str):
        f = (DIST / path.lstrip("/")).resolve()
        if not f.is_relative_to(DIST) or not f.is_file():
            f = DIST / "index.html"
        if not f.exists():
            self._head(503, "text/plain")
            self.wfile.write(b"UI not built: cd demo_ui/web && pnpm install && pnpm build")
            return
        self._head(200, mimetypes.guess_type(f.name)[0] or "application/octet-stream")
        self.wfile.write(f.read_bytes())


def calib_overlay(calib_path: Path, image_size: tuple[int, int]) -> dict:
    """table_calib.json (CONTRACT §6) -> workspace sector and tape crosses in normalised image coords,
    for bus.publish(calib=...). Needs numpy; the homography is fitted table -> image from the 4 crosses."""
    import numpy as np

    c = json.loads(Path(calib_path).read_text())
    src, dst = np.float64(c["table_cm"]), np.float64(c["image_px"])  # metres, pixels
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    H = np.linalg.svd(np.float64(A))[2][-1].reshape(3, 3)

    def project(pts):
        p = np.c_[pts, np.ones(len(pts))] @ H.T
        return (p[:, :2] / p[:, 2:] / np.float64(image_size)).round(4).tolist()

    b = np.radians(np.linspace(-34, 34, 24))
    outer = np.c_[0.28 * np.sin(b), -0.28 * np.cos(b)]
    inner = np.c_[0.12 * np.sin(b[::-1]), -0.12 * np.cos(b[::-1])]
    return {"workspace_px": project(np.vstack([outer, inner])), "crosses_px": project(src)}


def _plain(v):
    """numpy arrays / scalars -> JSON-friendly lists and floats, rounded so frames stay small."""
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    return v


# ---- replay -----------------------------------------------------------------------------------
def replay(bus: DemoBus, raw_dir: Path, onnx_dir: Path | None, n_eps=40, on_demand=False) -> None:
    """Loop over real sim episodes. Joints are sim radians shown as degrees (the joint map isn't
    fitted yet), gripper mapped to 0-100 per CONTRACT §6. The future chunk shown is the sim's own
    commanded plan; latency is the exported ONNX actually run on this Mac with the replayed inputs.
    on_demand: wait at idle for a "grab" from the UI before each episode, like the live runtime."""
    import numpy as np

    from policy.handover import GraspDetector, HandoverConfig, release_and_home

    chunk_file = sorted(raw_dir.glob("*.npz"))[0]
    d = np.load(chunk_file)
    idx = np.flatnonzero(d["success"])[:n_eps]
    steps = np.cumsum(d["phase_steps"])  # approach, descend, close, lift | handoff, hold
    pick_end, hand_end = int(steps[3]), int(steps[4])

    def to_ui(q):
        out = np.degrees(q).astype(np.float32)
        out[..., 5] = (q[..., 5] + 0.1745) / (1.745 + 0.1745) * 100
        return out

    sessions, k = {}, None
    if onnx_dir and onnx_dir.exists():
        try:
            import onnxruntime as ort

            for f in sorted(onnx_dir.glob("policy_k*_int8.onnx")):
                kk = int(f.stem.split("_")[1][1:])
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                sessions[kk] = ort.InferenceSession(str(f), so, providers=["CPUExecutionProvider"])
            k = max(sessions) if sessions else None
        except ImportError:
            pass

    state = {"k": k, "restart": False, "paused": False, "go": False}

    def on_cmd(c):
        if c.get("cmd") == "loops" and c.get("k") in sessions:
            state["k"] = c["k"]
        elif c.get("cmd") in ("grab", "start"):
            state["go"], state["restart"], state["paused"] = True, True, False
        elif c.get("cmd") == "estop":
            state["paused"] = True

    bus.on_command(on_cmd)
    bus.publish(
        mode="replay",
        source=f"Isaac Sim · {raw_dir.name}/{chunk_file.name}",
        links={"camera": False, "follower": False, "board": False},
        infer={"available_loops": sorted(sessions), "host": "mac" if sessions else None},
    )
    bus.log(f"replay: {len(idx)} successful sim episodes from {chunk_file.name}")
    if sessions:
        bus.log(f"timing exported INT8 policy on this Mac, K ∈ {sorted(sessions)}")

    tick = 1 / float(d["fps"])
    ep_n = 0
    while True:
        i = int(idx[ep_n % len(idx)])
        ep_n += 1
        T = int(d["length"][i])
        q, a = to_ui(d["obs_joint_pos"][i, :T]), to_ui(d["action"][i, :T])
        x, y, _, yaw = (float(v) for v in d["obs_box"][i, 3])
        box = {"xy": [x, y], "theta_deg": float(np.degrees(yaw)) % 180, "latched": True, "size_m": [0.089, 0.066]}
        env = np.float32([[x, y, np.cos(2 * yaw), np.sin(2 * yaw), 0.9, 0.57]])
        bus.publish(phase="idle", episode=ep_n, state=q[0], action=a[0], chunk=a[:30], box=box, grasp_err=0.0)
        bus.log(f"episode {ep_n}: box at r={np.hypot(x, y):.3f} m, bearing {np.degrees(np.arctan2(x, -y)):+.0f}°")
        if on_demand:
            while not state["go"]:
                time.sleep(0.05)
        else:
            time.sleep(1.2)
        state["go"] = state["restart"] = False
        # sim pick + scripted handoff + hold, then the real release/home script from handover.py
        tail = release_and_home(q[-1], HandoverConfig(home=tuple(q[0])))
        n_release = round(1.1 / tick)
        timeline = [("pick" if t < pick_end else "lift" if t < hand_end else "hold", q[t], a[t], a[t : t + 30])
                    for t in range(T)]
        timeline += [("release" if j < n_release else "home", r, r, tail[j : j + 30]) for j, r in enumerate(tail)]
        det, grasped, lat = GraspDetector(), False, deque(maxlen=90)
        last_phase, t_phase0 = None, time.time()
        for phase, qs, act, chunk in timeline:
            t0 = time.time()
            while state["paused"] and not state["restart"]:
                bus.publish(phase="estop")
                time.sleep(0.1)
            if state["restart"]:
                state["restart"] = False
                break
            if phase != last_phase:
                t_phase0, last_phase = time.time(), phase
                if phase != "pick":
                    bus.log({"lift": "scripted handover: 45° up-and-back, min-jerk 2.5 s",
                             "hold": "holding · waiting for a tug (or Enter)",
                             "release": "tug → opening gripper",
                             "home": "returning to rest"}[phase], "event")
            g_err = float(abs(act[5] - qs[5]))
            if not grasped and det.update(float(act[5]), float(qs[5])):
                grasped = True
                bus.log(f"grasp detected · jaws stalled {g_err:.0f} units off command", "event")
            infer = {}
            if sessions and phase == "pick":
                t1 = time.perf_counter()
                sessions[state["k"]].run(None, {"state": qs[None].astype(np.float32), "env": env})
                lat.append((time.perf_counter() - t1) * 1e3)
                infer = {"ms": lat[-1], "ms_hist": list(lat), "loops": state["k"],
                         "model": f"looped_k{state['k']}_int8", "hz": 1 / tick}
            bus.publish(phase=phase, state=qs, action=act, chunk=chunk, grasp_err=g_err, grasped=grasped,
                        phase_t=time.time() - t_phase0, **({"infer": infer} if infer else {}))
            time.sleep(max(0.0, tick - (time.time() - t0)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--replay", action="store_true")
    p.add_argument("--on-demand", action="store_true", help="replay waits for the UI's grab button")
    p.add_argument("--sim", type=Path, default=HERE.parent / "sim_data/raw/sim_demos_v1")
    p.add_argument("--onnx", type=Path, default=HERE.parent / "runs/smoke_looped/onnx")
    args = p.parse_args()
    bus = DemoBus(args.port).start()
    if args.replay:
        replay(bus, args.sim, args.onnx, on_demand=args.on_demand)
    else:
        bus.log("bridge up · waiting for the runtime to publish")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
