import { BOX_SIZE, CROSSES, WORKSPACE } from "../config";
import type { Telemetry } from "../types";

// Top-down plan in the robot base frame (CONTRACT §4), drawn from behind the arm:
// forward (−y) points up the screen and the robot's left (+x) is screen-left. Units: mm.
const W = 480;
const H = 360;
const X = (x: number) => W / 2 - x * 1000;
const Y = (y: number) => 14 + (y + 0.31) * 1000;
const polar = (r: number, deg: number) => {
  const a = (deg * Math.PI) / 180;
  return [X(r * Math.sin(a)), Y(-r * Math.cos(a))] as const;
};

function arc(r: number, a0: number, a1: number) {
  const [x0, y0] = polar(r, a0);
  const [x1, y1] = polar(r, a1);
  return `M${x0},${y0} A${r * 1000},${r * 1000} 0 0 ${a0 < a1 ? 0 : 1} ${x1},${y1}`;
}

export function inReach(tele: Telemetry) {
  const xy = tele.box?.xy;
  if (!xy) return null;
  const { rMin, rMax, bearing } = WORKSPACE;
  const r = Math.hypot(...xy);
  const brg = (Math.atan2(xy[0], -xy[1]) * 180) / Math.PI;
  return r >= rMin && r <= rMax && Math.abs(brg) <= bearing;
}

export default function TablePlan({ tele }: { tele: Telemetry }) {
  const reach = inReach(tele);
  return (
    <section className="panel plan">
      <div className="panel-head">
        <span className="label">
          Table frame
        </span>
        {reach !== null && (
          <span className="label" style={{ color: reach ? "var(--ok)" : "var(--warn)" }}>
            {reach ? "in reach" : "outside workspace"}
          </span>
        )}
      </div>
      <PlanSvg tele={tele} />
    </section>
  );
}

export function PlanSvg({ tele }: { tele: Telemetry }) {
  const { rMin, rMax, bearing: b } = WORKSPACE;
  const sector = `${arc(rMax, -b, b)} L${polar(rMin, b).join(",")} ${arc(rMin, b, -b).replace("M", "L")} Z`;
  const box = tele.box;
  const pan = tele.state?.[0];
  const panBearing = pan !== undefined ? pan * (tele.pan_bearing_sign ?? -1) : null;

  let boxPath = "";
  if (box?.xy) {
    const [cx, cy] = box.xy;
    const t = ((box.theta_deg ?? 0) * Math.PI) / 180;
    const [L, Wd] = box.size_m ?? BOX_SIZE;
    boxPath = [
      [L / 2, Wd / 2],
      [-L / 2, Wd / 2],
      [-L / 2, -Wd / 2],
      [L / 2, -Wd / 2],
    ]
      .map(([u, v], i) => `${i ? "L" : "M"}${X(cx + Math.cos(t) * u - Math.sin(t) * v)},${Y(cy + Math.sin(t) * u + Math.cos(t) * v)}`)
      .join("") + "Z";
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
        {[0.1, 0.2, 0.3].map((r) => (
          <g key={r}>
            <path d={arc(r, -50, 50)} fill="none" stroke="var(--line-2)" strokeDasharray="1 4" />
            <text x={polar(r, -50)[0] + 6} y={polar(r, -50)[1] + 4} fill="var(--faint)" fontSize={14}>
              {r * 100} cm
            </text>
          </g>
        ))}
        <path d={sector} fill="var(--cool-soft)" stroke="var(--cool)" strokeOpacity={0.5} />
        <line x1={X(0)} y1={Y(0)} x2={polar(0.32, 0)[0]} y2={polar(0.32, 0)[1]} stroke="var(--line-2)" />
        {CROSSES.map((c) => (
          <g key={c.id} stroke="var(--faint)">
            <line x1={X(c.x) - 6} x2={X(c.x) + 6} y1={Y(c.y)} y2={Y(c.y)} />
            <line x1={X(c.x)} x2={X(c.x)} y1={Y(c.y) - 6} y2={Y(c.y) + 6} />
            <text x={X(c.x) + 8} y={Y(c.y) - 6} fill="var(--faint)" stroke="none" fontSize={14}>
              {c.id}
            </text>
          </g>
        ))}
        {panBearing !== null && (
          <g>
            <line
              x1={X(0)}
              y1={Y(0)}
              x2={polar(rMax + 0.03, panBearing)[0]}
              y2={polar(rMax + 0.03, panBearing)[1]}
              stroke="var(--text)"
              strokeWidth={1.5}
            />
            <circle cx={polar(rMax + 0.03, panBearing)[0]} cy={polar(rMax + 0.03, panBearing)[1]} r={3} fill="var(--text)" />
          </g>
        )}
        {boxPath && <path d={boxPath} fill="var(--signal-soft)" stroke="var(--signal)" strokeWidth={1.5} />}
        <path d={`M${X(0.045)},${Y(0)} A45,45 0 0 1 ${X(-0.045)},${Y(0)} Z`} fill="var(--raise)" stroke="var(--line-2)" />
        <text x={X(0)} y={Y(0) - 12} textAnchor="middle" fill="var(--faint)" fontSize={14}>
          Base
        </text>
        {box?.xy && (
          <text x={12} y={H - 10} fill="var(--dim)" fontSize={15}>
            x {(box.xy[0] * 100).toFixed(1)} · y {(box.xy[1] * 100).toFixed(1)} cm · θ {(box.theta_deg ?? 0).toFixed(0)}°
          </text>
        )}
    </svg>
  );
}
