import { CONTROL_HZ } from "../config";
import type { Scorecard, Telemetry } from "../types";
import Sparkline from "./Sparkline";

const C = 98; // ring centre
const pct = (xs: number[], p: number) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
};
const pol = (r: number, deg: number) => {
  const a = ((deg - 90) * Math.PI) / 180;
  return [C + r * Math.cos(a), C + r * Math.sin(a)];
};
const arc = (r: number, a0: number, a1: number) => {
  const [x0, y0] = pol(r, a0);
  const [x1, y1] = pol(r, a1);
  return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1},${y1}`;
};

interface Props {
  tele: Telemetry;
  card: Scorecard | null;
  onLoops: (k: number) => void;
}

export default function InferencePanel({ tele, card, onLoops }: Props) {
  const inf = tele.infer ?? {};
  const hist = inf.ms_hist ?? [];
  const p50 = pct(hist, 0.5);
  const p95 = pct(hist, 0.95);
  const K = inf.loops ?? 0;
  const active = tele.phase === "pick" && inf.ms !== undefined;
  const budget = 1000 / CONTROL_HZ;
  const avail = inf.available_loops;
  const host = inf.host === "uno-q" ? "on UNO Q" : inf.host === "mac" ? "on Mac · replay" : "no inference yet";
  const gap = K > 1 ? 7 : 0;
  const cycle = 1.2;

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          Policy inference
        </span>
        <span className="label" style={{ color: inf.host === "uno-q" ? "var(--signal)" : "var(--faint)" }}>
          {host}
        </span>
      </div>
      <div className="infer">
        <div className="ring-wrap">
          <svg width={196} height={196} viewBox="0 0 196 196">
            {/* outer: the 30-step action chunk, consumed at 30 Hz */}
            {Array.from({ length: 30 }, (_, i) => {
              const [x0, y0] = pol(90, i * 12);
              const [x1, y1] = pol(i % 5 ? 85 : 81, i * 12);
              return <line key={i} x1={x0} y1={y0} x2={x1} y2={y1} stroke="var(--line-2)" />;
            })}
            {active && (
              <g className="sweep">
                <line x1={C} y1={C - 92} x2={C} y2={C - 78} stroke="var(--signal)" strokeWidth={2} />
              </g>
            )}
            {/* inner: one shared block, applied K times */}
            {K > 0 ? (
              Array.from({ length: K }, (_, i) => (
                <path
                  key={`${K}-${i}`}
                  className={`ring-seg ${active ? "live" : ""}`}
                  d={arc(66, (i * 360) / K + gap / 2, ((i + 1) * 360) / K - gap / 2 - (K === 1 ? 0.01 : 0))}
                  style={{ ["--cycle" as string]: `${cycle}s`, animationDelay: `${(i * cycle) / K}s` }}
                />
              ))
            ) : (
              <circle cx={C} cy={C} r={66} fill="none" stroke="var(--line)" strokeWidth={10} />
            )}
            <text x={C} y={C + 2} textAnchor="middle" fontWeight={600} fontSize={34} fill="var(--text)">
              {inf.ms !== undefined ? (p50 ?? inf.ms).toFixed(inf.ms < 10 ? 2 : 1) : "—"}
            </text>
            <text x={C} y={C + 22} textAnchor="middle" fontSize={11} fill="var(--faint)">
              ms per chunk
            </text>
          </svg>
          <div className="stats">
            <Stat k="Loops K" v={K || "—"} />
            <Stat k="p95" v={p95 !== null ? p95.toFixed(2) : "—"} u="ms" />
            <Stat k="Headroom" v={p50 ? `${Math.floor(budget / p50)}×` : "—"} u={`/ ${budget.toFixed(0)} ms`} />
            <Stat k="Model" v={<span style={{ fontSize: 12 }}>{inf.model ?? "—"}</span>} />
          </div>
        </div>

        <div className="subhead">
          <span className="label">Latency, last {hist.length || 0} steps</span>
          <span className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>
            {tele.board?.power_w !== undefined ? `${tele.board.power_w.toFixed(2)} W · ` : ""}
            {tele.board?.ram_mb !== undefined ? `${tele.board.ram_mb.toFixed(0)} MB RSS` : ""}
          </span>
        </div>
        <div style={{ marginTop: 8 }}>
          <Sparkline values={hist} w={418} h={34} min={0} stroke="var(--signal)" />
        </div>

        <div className="subhead">
          <span className="label">Loops: accuracy vs compute</span>
        </div>
        <div className="kpick">
          {Array.from({ length: 8 }, (_, i) => i + 1).map((k) => (
            <button key={k} className={k === K ? "on" : ""} disabled={avail ? !avail.includes(k) : false} onClick={() => onLoops(k)}>
              {k}
            </button>
          ))}
        </div>
        {card?.loop_curve && <LoopCurve points={card.loop_curve.points} k={K} source={card.loop_curve.source} />}
      </div>
    </section>
  );
}

function Stat({ k, v, u }: { k: string; v: React.ReactNode; u?: string }) {
  return (
    <div className="stat">
      <div className="label">{k}</div>
      <div className="v">
        {v}
        {u && <small>{u}</small>}
      </div>
    </div>
  );
}

function LoopCurve({ points, k, source }: { points: Record<string, number>; k: number; source: string }) {
  const ks = Object.keys(points).map(Number).filter((x) => x <= 8).sort((a, b) => a - b);
  const vs = ks.map((x) => points[x]);
  const w = 418;
  const h = 44;
  const lo = Math.min(...vs);
  const hi = Math.max(...vs);
  const X = (x: number) => ((x - 0.5) / 8) * w; // centred under the K buttons
  const Y = (v: number) => 6 + (1 - (v - lo) / (hi - lo || 1)) * (h - 14);
  const best = ks[vs.indexOf(lo)];
  return (
    <div style={{ marginTop: 10 }}>
      <svg width={w} height={h} style={{ display: "block" }}>
        <path d={ks.map((x, i) => `${i ? "L" : "M"}${X(x)},${Y(vs[i])}`).join("")} fill="none" stroke="var(--dim)" strokeWidth={1.25} />
        {ks.map((x, i) => (
          <circle key={x} cx={X(x)} cy={Y(vs[i])} r={x === k ? 4 : 2} fill={x === k ? "var(--signal)" : x === best ? "var(--text)" : "var(--faint)"} />
        ))}
      </svg>
      <div className="mono" style={{ fontSize: 10, color: "var(--faint)", marginTop: 6 }}>
        {source} · best K={best}
      </div>
    </div>
  );
}
