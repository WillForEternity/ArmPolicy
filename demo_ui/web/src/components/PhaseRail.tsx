import { PHASES } from "../config";
import type { Telemetry } from "../types";

// Expected durations (s) for the scripted parts, from handover.HandoverConfig; pick is open-ended.
const EXPECT: Record<string, number> = { pick: 5, lift: 2.5, hold: 3, release: 1.1, home: 3 };

export default function PhaseRail({ tele }: { tele: Telemetry }) {
  const phase = tele.phase ?? "idle";
  const at = PHASES.findIndex((p) => p.id === phase);
  const prog = Math.min(1, (tele.phase_t ?? 0) / (EXPECT[phase] ?? 3));
  return (
    <section className={`panel rail ${phase === "estop" ? "estop-rail" : ""}`}>
      <div className="rail-legend">
        <span className="label">
          Sequence
        </span>
        <div className="legend label">
          <span>
            <span className="sw" style={{ background: "var(--signal)" }} />
            Learned · runs on UNO Q
          </span>
          <span>
            <span className="sw" style={{ background: "var(--cool)" }} />
            Scripted · Mac
          </span>
        </div>
      </div>
      <div className="rail-track">
        {PHASES.map((p, i) => {
          const cls = [p.who, i < at ? "done" : "", i === at ? "now" : ""].join(" ");
          return (
            <div key={p.id} className={`seg ${cls}`} style={{ flex: p.weight }}>
              <div className="fill" style={i === at ? { transform: `scaleX(${prog})` } : undefined} />
              <span className="nm">{p.label}</span>
              <span className="who">
                {i === at && tele.phase_t !== undefined ? `${tele.phase_t.toFixed(1)} s` : p.who === "learned" ? "policy → 30-step chunks" : hint(p.id)}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function hint(id: string) {
  return { lift: "45° up & back", hold: "wait for tug", release: "open jaws", home: "rest pose" }[id] ?? "";
}
