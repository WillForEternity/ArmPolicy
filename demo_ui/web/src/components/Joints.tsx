import { JOINTS } from "../config";
import type { Telemetry } from "../types";
import Sparkline from "./Sparkline";

export default function Joints({ tele, history }: { tele: Telemetry; history: number[][] }) {
  const q = tele.state;
  const a = tele.action;
  const chunk = tele.chunk ?? [];
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          Follower joints
        </span>
        <span>
          <span className="key"><i style={{ background: "var(--text)" }} />Measured</span>
          <span className="key"><i style={{ background: "var(--signal)" }} />Target</span>
          <span className="key"><i style={{ background: "var(--signal)", opacity: 0.35 }} />Next second</span>
        </span>
      </div>
      <div className="joints">
        {JOINTS.map((j, i) => {
          const [lo, hi] = j.range;
          const x = (v: number) => ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 1000;
          const zero = x(lo < 0 ? 0 : lo);
          const step = j.key === "gripper" ? 10 : 30;
          const ticks = [];
          for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(t);
          const grip = j.key === "gripper";
          return (
            <div className="jrow" key={j.key}>
              <div>
                <div className="label" style={{ color: "var(--text)" }}>
                  {j.label}
                </div>
                {grip && tele.grasped && <span className="stall" style={{ marginLeft: 0 }}>HELD</span>}
              </div>
              <svg width="100%" height={30} viewBox="0 0 1000 30" preserveAspectRatio="none">
                <line x1={0} x2={1000} y1={15} y2={15} stroke="var(--line-2)" vectorEffect="non-scaling-stroke" />
                {ticks.map((t) => (
                  <line key={t} x1={x(t)} x2={x(t)} y1={t === 0 ? 8 : 12} y2={t === 0 ? 22 : 18} stroke="var(--faint)" vectorEffect="non-scaling-stroke" />
                ))}
                {q && <rect x={Math.min(zero, x(q[i]))} y={12} width={Math.abs(x(q[i]) - zero)} height={6} fill="var(--text)" opacity={0.14} />}
                {chunk.map((c, k) => (
                  <line
                    key={k}
                    x1={x(c[i])}
                    x2={x(c[i])}
                    y1={19}
                    y2={28}
                    stroke="var(--signal)"
                    strokeOpacity={0.55 * (1 - k / chunk.length) + 0.08}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
                {a && <line x1={x(a[i])} x2={x(a[i])} y1={1} y2={13} stroke="var(--signal)" strokeWidth={3} vectorEffect="non-scaling-stroke" />}
                {q && <line x1={x(q[i])} x2={x(q[i])} y1={4} y2={26} stroke="var(--text)" strokeWidth={2} vectorEffect="non-scaling-stroke" />}
              </svg>
              <div className="v">
                {q ? q[i].toFixed(1) : "—"}
                {a && <small>→ {a[i].toFixed(1)}</small>}
              </div>
              <Sparkline values={history.map((h) => h[i])} w={72} h={26} min={grip ? 0 : undefined} max={grip ? 100 : undefined} />
            </div>
          );
        })}
      </div>
    </section>
  );
}
