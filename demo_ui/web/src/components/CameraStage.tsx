import { useEffect, useRef, useState } from "react";
import { BOX_SIZE } from "../config";
import { useCamera } from "../hooks";
import { PlanSvg } from "./TablePlan";
import BridgeFeed from "./BridgeFeed";
import type { Telemetry, XY } from "../types";

const PHASE_WORD: Record<string, string> = {
  idle: "Ready",
  pick: "Picking",
  lift: "Handing over",
  hold: "Take it",
  release: "Releasing",
  home: "Returning",
  estop: "Stopped",
};

export default function CameraStage({ tele }: { tele: Telemetry }) {
  const cam = useCamera(tele);
  const video = useRef<HTMLVideoElement>(null);
  const [dims, setDims] = useState<XY>([1920, 1080]);
  const [fps, setFps] = useState<number | null>(null);

  useEffect(() => {
    const v = video.current;
    if (!v) return;
    v.srcObject = cam.stream;
    const s = cam.stream?.getVideoTracks()[0]?.getSettings();
    setFps(s?.frameRate ? Math.round(s.frameRate) : null);
  }, [cam.stream]);

  const [W, H] = dims;
  const box = tele.box;
  const quad = box?.corners_px?.map(([u, v]) => [u * W, v * H] as XY);
  const ws = tele.calib?.workspace_px?.map(([u, v]) => [u * W, v * H] as XY);
  const crosses = tele.calib?.crosses_px?.map(([u, v]) => [u * W, v * H] as XY);
  const r = box?.xy ? Math.hypot(box.xy[0], box.xy[1]) : null;
  const bridgeFeed = cam.deviceId === "bridge";
  const hasFeed = !!cam.stream || bridgeFeed;
  const current = cam.devices.find((d) => d.deviceId === cam.deviceId);
  const phase = tele.phase ?? "idle";
  const s = Math.max(W, H) / 1000; // overlay stroke scale

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          Overhead
          {current && <span style={{ color: "var(--faint)", marginLeft: 10 }}>{current.label.replace(/\s*\(.*\)/, "")}</span>}
        </span>
        <div className="meta">
          {hasFeed && !bridgeFeed && (
            <span className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
              {W}×{H}
              {fps ? ` · ${fps} fps` : ""}
            </span>
          )}
          <select className="select" value={cam.deviceId ?? ""} onChange={(e) => cam.choose(e.target.value)}>
            {!cam.deviceId && <option value="">choose camera…</option>}
            {cam.devices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || "camera"}
              </option>
            ))}
            <option value="bridge">bridge stream (annotated)</option>
            <option value="none">off</option>
          </select>
        </div>
      </div>
      <div className="stage">
        {bridgeFeed ? (
          <BridgeFeed onSize={(w, h) => (w !== dims[0] || h !== dims[1] ? setDims([w, h]) : undefined)} />
        ) : (
          <video
            ref={video}
            autoPlay
            muted
            playsInline
            onLoadedMetadata={(e) => setDims([e.currentTarget.videoWidth, e.currentTarget.videoHeight])}
          />
        )}
        {!hasFeed && (
          <div className="nofeed">
            <PlanSvg tele={tele} />
            <div className="nofeed-note">
              <div className="label">No camera, showing the table</div>
              <p>{cam.error ?? (cam.deviceId === "none" ? "camera off" : "waiting for camera permission")}</p>
            </div>
          </div>
        )}
        {hasFeed && !bridgeFeed && (
          <svg className="overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
            {ws && (
              <polygon
                points={ws.map((p) => p.join(",")).join(" ")}
                fill="rgba(167,182,196,0.06)"
                stroke="var(--cool)"
                strokeWidth={1.2 * s}
                strokeDasharray={`${6 * s} ${5 * s}`}
              />
            )}
            {crosses?.map(([x, y], i) => (
              <g key={i} stroke="var(--cool)" strokeWidth={1.2 * s} opacity={0.7}>
                <line x1={x - 9 * s} x2={x + 9 * s} y1={y} y2={y} />
                <line x1={x} x2={x} y1={y - 9 * s} y2={y + 9 * s} />
              </g>
            ))}
            {quad && quad.length === 4 && <BoxMark quad={quad} s={s} r={r} theta={box?.theta_deg} latched={box?.latched} />}
          </svg>
        )}
        <div className="hud">
          <div>
            <div className="label" style={{ marginBottom: 8 }}>
              {tele.mode === "replay" ? `Sim episode ${tele.episode ?? "–"}` : "State"}
            </div>
            <div className={`big ${phase === "pick" ? "learned" : ""}`}>{PHASE_WORD[phase] ?? phase}</div>
          </div>
          {box?.xy && (
            <div className="mono" style={{ fontSize: 12, color: "var(--dim)", lineHeight: 1.6, paddingBottom: 2 }}>
              Box {(r! * 100).toFixed(1)} cm away, {(box.theta_deg ?? 0).toFixed(0)}°
              <br />
              {(BOX_SIZE[0] * 1000).toFixed(0)}×{(BOX_SIZE[1] * 1000).toFixed(0)} mm · {box.latched ? "pose locked" : "tracking"}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function BoxMark({ quad, s, r, theta, latched }: { quad: XY[]; s: number; r: number | null; theta?: number; latched?: boolean }) {
  const tick = 16 * s;
  const corners = quad.map((p, i) => {
    const a = quad[(i + 3) % 4];
    const b = quad[(i + 1) % 4];
    const toward = (q: XY) => {
      const dx = q[0] - p[0];
      const dy = q[1] - p[1];
      const n = Math.hypot(dx, dy) || 1;
      return [p[0] + (dx / n) * tick, p[1] + (dy / n) * tick];
    };
    const [ax, ay] = toward(a);
    const [bx, by] = toward(b);
    return `M${ax},${ay}L${p[0]},${p[1]}L${bx},${by}`;
  });
  const top = quad.reduce((m, p) => (p[1] < m[1] ? p : m), quad[0]);
  return (
    <g>
      <polygon points={quad.map((p) => p.join(",")).join(" ")} fill="rgba(255,90,31,0.08)" stroke="var(--signal)" strokeWidth={1 * s} opacity={0.7} />
      <path d={corners.join("")} fill="none" stroke="var(--signal)" strokeWidth={3 * s} />
      <g transform={`translate(${top[0]},${top[1] - 14 * s})`} fontSize={15 * s}>
        <rect x={-10 * s} y={-19 * s} width={(latched ? 196 : 150) * s} height={28 * s} rx={14 * s} fill="var(--signal)" />
        <text fill="#fff" fontWeight={600}>
          Box {r ? `· ${(r * 100).toFixed(1)} cm` : ""} {theta !== undefined ? `· ${theta.toFixed(0)}°` : ""} {latched ? "· locked" : ""}
        </text>
      </g>
    </g>
  );
}
