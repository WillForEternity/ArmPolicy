import { useClock, type Link } from "../hooks";
import type { Telemetry } from "../types";

interface Props {
  tele: Telemetry;
  link: Link;
  onStart: () => void;
  onStop: () => void;
  onScore: () => void;
  onDemo: () => void;
}

function Chip({ name, state }: { name: string; state: "on" | "off" | "warn" }) {
  return (
    <span className="chip">
      <span className={`dot ${state}`} />
      <span className="label" style={{ color: state === "on" ? "var(--text)" : undefined }}>
        {name}
      </span>
    </span>
  );
}

export default function TopBar({ tele, link, onStart, onStop, onScore, onDemo }: Props) {
  const now = useClock();
  const l = tele.links ?? {};
  const bridge = link === "open" ? "on" : link === "stale" || link === "connecting" ? "warn" : "off";
  const replay = tele.mode === "replay";
  return (
    <header className="top">
      <div className="brand">
        <b>Pick &amp; handover</b>
        <span className="label">SO-101 arm · Arduino UNO Q</span>
      </div>
      <div className="links">
        <Chip name="Bridge" state={bridge} />
        <Chip name="Camera" state={l.camera ? "on" : "off"} />
        <Chip name="Follower" state={l.follower ? "on" : "off"} />
        <Chip name="UNO Q" state={l.board ? "on" : "off"} />
      </div>
      <span className={`mode ${tele.mode === "live" && link === "open" ? "live" : ""}`}>
        {link !== "open" ? "Offline" : !tele.mode ? "Connecting" : replay ? "Sim replay" : "Live"}
      </span>
      <span className="clock">{now.toLocaleTimeString([], { hour12: false })}</span>
      <button className="btn" onClick={onDemo}>
        Demo
      </button>
      <button className="btn" onClick={onScore}>
        Scorecard
      </button>
      <button className="btn go" onClick={onStart}>
        Run
      </button>
      <button className="btn stop" onClick={onStop}>
        Stop
      </button>
    </header>
  );
}
