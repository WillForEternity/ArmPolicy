import type { LogEntry } from "../types";

const fmt = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour12: false });

export default function LogFeed({ logs, source }: { logs: LogEntry[]; source?: string }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          Events
        </span>
        {source && <span className="mono" style={{ fontSize: 11, color: "var(--faint)" }}>{source}</span>}
      </div>
      <div className="log">
        {logs.slice(-9).map((l, i) => (
          <div key={`${l.t}-${i}`} className={`row ${l.level}`}>
            <time>{fmt(l.t)}</time>
            <span>{l.text}</span>
          </div>
        ))}
        {!logs.length && <div className="row">no events yet</div>}
      </div>
    </section>
  );
}
