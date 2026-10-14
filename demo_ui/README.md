# Demo UI

One screen for the live demo: overhead camera with the box overlay, the pick/handover sequence, policy
inference on the UNO Q (latency, loops K), follower joints with the 1 s action chunk, and an efficiency
scorecard for the judges.

```
cd ArmPolicy
.venv/bin/python -m demo_ui.bridge --replay --on-demand   # no hardware: Isaac Sim episodes, one per grab
open http://localhost:8765/#demo
```

Two tabs. **Demo** (`#demo`, key `D`) is for the audience: the camera full screen and one button,
"Grab me the Arduino Q", which sends `{"cmd": "grab"}`. While a run is going it turns into Stop.
**Console** is the engineering view described below.

Build once after UI changes: `cd demo_ui/web && pnpm install && pnpm build`. For UI work, run `pnpm dev`
(port 5173, proxies to the bridge).

**Keys:** `Space` grab · `D` demo/console · `Esc` stop · `Enter` release (backup to the tug) · `S` scorecard · `1`–`8` loops K.
Open `http://localhost:8765/#scorecard` to start on the scorecard.

The Stop button only sends `{"cmd": "estop"}` to the runtime. It does not replace the arm's power switch.

## Wiring the live runtime

The bridge is standard library only and never touches hardware. The runtime pushes what it knows each
tick, and reads UI commands back:

```python
from demo_ui.bridge import DemoBus, calib_overlay

bus = DemoBus().start()
bus.publish(mode="live", calib=calib_overlay("table_calib.json", (1920, 1080)),
            pan_bearing_sign=+1)          # sign of LeRobot pan vs. table bearing, from the joint map
# every control tick:
bus.publish(phase="pick",                 # idle | pick | lift | hold | release | home | estop
            state=q, action=a, chunk=chunk,                       # LeRobot degrees, gripper 0-100
            box={"xy": [x, y], "theta_deg": th, "corners_px": quad_norm, "latched": held},
            infer={"host": "uno-q", "ms": ms, "ms_hist": last_90, "loops": k, "model": "looped_k4_int8"},
            board={"ram_mb": rss, "power_w": w},
            links={"camera": True, "follower": True, "board": True},
            grasp_err=err, grasped=det_fired, phase_t=seconds_in_phase)
bus.log("grasp detected", "event")
for cmd in bus.poll_commands():           # {"cmd": "grab" | "estop" | "release" | "loops", "k": 4}
    ...
```

`corners_px` are the box corners from `perception.detect`, divided by the frame width and height.

**Camera:** if the runtime owns the iPhone (OpenCV), call `bus.publish_frame(jpeg_bytes)` with
annotated frames. The first frame sets `video=True`, and the UI then shows `/frame.jpg` instead of
opening the camera itself. Without frames, the browser opens the iPhone (Continuity Camera) directly.

**Recording demos** (`scripts/record_ui.py`, port 8766): publish `task="record"` and `saved=<n>`. The
Demo tab becomes the collection screen: Start recording (`grab`), Save (`release`), Discard (`estop`),
with `phase` pick = recording, home = saving.

## Scorecard

`scorecard.json` is re-read every time the sheet opens. Leave unmeasured values `null` (shown as
"pending"), and set `on` to where the number was measured. Only `uno-q` is highlighted as on-device.
