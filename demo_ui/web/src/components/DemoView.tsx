import { useEffect, useRef, useState } from "react";
import { sendCmd, useCamera, type Link } from "../hooks";
import BridgeFeed from "./BridgeFeed";
import type { Telemetry, XY } from "../types";

// What the person at the table should read, per phase.
const SAY: Record<string, [string, string]> = {
  idle: ["Ready when you are", "Put the box anywhere in front of the arm."],
  pick: ["Reaching for it", "A small looped transformer is steering the arm."],
  lift: ["Got it", "Bringing it up to you."],
  hold: ["Here you go", "Give it a gentle tug and I'll let go."],
  release: ["Letting go", ""],
  home: ["All yours", "Heading back to rest."],
  estop: ["Stopped", "Tap the button to try again."],
};

// Rough share of the whole run each phase takes, for the progress line.
const SPAN: Record<string, [number, number, number]> = {
  pick: [0, 0.5, 5],
  lift: [0.5, 0.68, 2.5],
  hold: [0.68, 0.8, 3],
  release: [0.8, 0.86, 1.1],
  home: [0.86, 1, 3],
};

interface Props {
  tele: Telemetry;
  link: Link;
  onConsole: () => void;
}

export default function DemoView({ tele, link, onConsole }: Props) {
  const cam = useCamera(tele);
  const video = useRef<HTMLVideoElement>(null);
  const [dims, setDims] = useState<XY>([1920, 1080]);
  const phase = tele.phase ?? "idle";
  const busy = phase !== "idle" && phase !== "estop";
  const live = tele.mode === "live";
  const quad = tele.box?.corners_px;
  const seesBox = !live || (quad?.length ?? 0) === 4;
  const offline = link === "down";
  let [title, sub] = offline ? ["Not connected", "Start the bridge on the Mac."] : SAY[phase] ?? SAY.idle;
  if (phase === "pick" && tele.infer?.host === "uno-q") sub = "A small looped transformer on the Arduino UNO Q is steering the arm.";

  useEffect(() => {
    if (video.current) video.current.srcObject = cam.stream;
  }, [cam.stream]);

  const span = SPAN[phase];
  const progress = span ? span[0] + (span[1] - span[0]) * Math.min(1, (tele.phase_t ?? 0) / span[2]) : 0;
  const [W, H] = dims;
  const ms = tele.infer?.ms;

  const press = () => sendCmd(busy ? { cmd: "estop" } : { cmd: "grab" });

  // Data collection: same screen, teleop instead of the policy. grab = start, release = save, estop = discard.
  const record = tele.task === "record";
  const recording = record && phase === "pick";
  const saving = record && phase === "home";
  const saved = tele.saved ?? 0;
  if (record) {
    title = offline ? title : recording ? "Recording" : saving ? "Saving" : "Ready to record";
    sub = offline
      ? sub
      : recording
        ? "Teleop the arm: pick the box and lift it. Save once it's up."
        : saving
          ? "Writing the episode."
          : `${saved} ${saved === 1 ? "demo" : "demos"} saved. Place the box somewhere new, then start.`;
  }
  const t = tele.phase_t ?? 0;
  const clock = `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  return (
    <div className="demo">
      {cam.deviceId === "bridge" ? (
        <BridgeFeed className="feed" />
      ) : (
        <video
          ref={video}
          className="feed"
          autoPlay
          muted
          playsInline
          onLoadedMetadata={(e) => setDims([e.currentTarget.videoWidth, e.currentTarget.videoHeight])}
        />
      )}
      {!cam.stream && cam.deviceId !== "bridge" && (
        <div className="demo-nocam">
          <p>{cam.error ? `Camera: ${cam.error}` : "Waiting for the camera…"}</p>
        </div>
      )}

      {quad && quad.length === 4 && cam.stream && (
        <svg className="demo-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice">
          <polygon
            className={busy ? "" : "breathe"}
            points={quad.map(([u, v]) => `${u * W},${v * H}`).join(" ")}
            fill="rgba(255,255,255,0.06)"
            stroke="#fff"
            strokeWidth={Math.max(W, H) / 600}
            strokeLinejoin="round"
          />
        </svg>
      )}

      <div className="demo-top">
        <button className="demo-link" onClick={onConsole}>
          Console
        </button>
        {recording && (
          <span className="rec">
            <i />
            {clock}
          </span>
        )}
        <span className="demo-where">
          {record ? "Collecting demos" : tele.mode === "replay" ? "Simulation replay" : tele.infer?.host === "uno-q" ? "Running on Arduino UNO Q" : ""}
          {ms !== undefined && phase === "pick" ? ` · ${ms.toFixed(ms < 10 ? 1 : 0)} ms per step` : ""}
        </span>
        <select className="demo-cam" value={cam.deviceId ?? ""} onChange={(e) => cam.choose(e.target.value)} aria-label="Camera">
          {!cam.deviceId && <option value="">Camera…</option>}
          {cam.devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || "Camera"}
            </option>
          ))}
          <option value="bridge">Mac perception stream</option>
        </select>
      </div>

      <div className="demo-bottom">
        <h1 key={title}>{title}</h1>
        {sub && <p>{sub}</p>}
        {record ? (
          <div className="grab-row">
            {recording ? (
              <>
                <button className="grab" onClick={() => sendCmd({ cmd: "release" })}>
                  Save
                </button>
                <button className="grab quiet" onClick={() => sendCmd({ cmd: "estop" })}>
                  Discard
                </button>
              </>
            ) : (
              <button className="grab" onClick={() => sendCmd({ cmd: "grab" })} disabled={offline || saving}>
                {saving ? "Saving…" : "Start recording"}
              </button>
            )}
          </div>
        ) : (
          <button
            className={`grab ${busy ? "stop" : ""}`}
            onClick={press}
            disabled={offline || (!busy && !seesBox)}
          >
            {busy ? "Stop" : seesBox ? "Grab me the Arduino Q" : "I can't see the box"}
          </button>
        )}
      </div>

      {!record && <div className="demo-progress" style={{ transform: `scaleX(${progress})`, opacity: busy ? 1 : 0 }} />}
    </div>
  );
}
