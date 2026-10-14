import type { Scorecard as Card, ScoreRow } from "../types";

const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toLocaleString() : Number.isInteger(v) ? String(v) : v.toFixed(2));

function delta(r: ScoreRow) {
  if (r.baseline === null || r.optimized === null || !r.baseline || !r.optimized) return "";
  const f = r.better === "lower" ? r.baseline / r.optimized : r.optimized / r.baseline;
  if (f >= 1.15) return `${f.toFixed(1)}× ${r.better === "lower" ? "lower" : "higher"}`;
  const pct = ((r.optimized - r.baseline) / r.baseline) * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

function Val({ v, unit }: { v: number | null; unit: string }) {
  if (v === null) return <td className="val pending">pending</td>;
  return (
    <td className="val">
      {fmt(v)}
      {unit && <small>{unit}</small>}
    </td>
  );
}

export default function Scorecard({ card, onClose }: { card: Card | null; onClose: () => void }) {
  return (
    <div className="sheet" onClick={onClose}>
      <div className="label">
        Efficiency scorecard
      </div>
      <h1>
        Same task. <span style={{ color: "var(--signal)" }}>One block,</span> looped.
      </h1>
      {!card ? (
        <p className="mono" style={{ color: "var(--faint)" }}>
          scorecard.json not reachable, is the bridge running?
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th />
              <th>
                <div className="label">Baseline</div>
                <div className="nm">{card.baseline.name}</div>
                <div className="note">{card.baseline.detail}</div>
              </th>
              <th className="opt">
                <div className="label">Optimized</div>
                <div className="nm">{card.optimized.name}</div>
                <div className="note">{card.optimized.detail}</div>
              </th>
              <th>
                <div className="label">Change</div>
              </th>
            </tr>
          </thead>
          <tbody>
            {card.rows.map((r) => (
              <tr key={r.metric}>
                <td className="metric">
                  {r.metric}
                  <span className={`tag ${r.on === "uno-q" ? "board" : ""}`}>{r.on === "uno-q" ? "UNO Q" : r.on}</span>
                  {r.note && <div className="note">{r.note}</div>}
                </td>
                <Val v={r.baseline} unit={r.unit} />
                <Val v={r.optimized} unit={r.unit} />
                <td className="delta">{delta(r)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div className="mono" style={{ marginTop: "auto", paddingTop: 32, fontSize: 11, color: "var(--faint)" }}>
        Press S or click anywhere to close. Tags show where each number was measured.
      </div>
    </div>
  );
}
