import { useEffect, useState } from "react";
import CameraStage from "./components/CameraStage";
import DemoView from "./components/DemoView";
import InferencePanel from "./components/InferencePanel";
import Joints from "./components/Joints";
import LogFeed from "./components/LogFeed";
import PhaseRail from "./components/PhaseRail";
import Scorecard from "./components/Scorecard";
import TablePlan from "./components/TablePlan";
import TopBar from "./components/TopBar";
import { sendCmd, useScorecard, useTelemetry } from "./hooks";

export default function App() {
  const { tele, logs, link, history } = useTelemetry();
  const [score, setScore] = useState(() => location.hash === "#scorecard");
  const [view, setView] = useState<"demo" | "console">(() => (location.hash === "#demo" ? "demo" : "console"));
  const go = (v: "demo" | "console") => {
    setView(v);
    window.history.replaceState(null, "", v === "demo" ? "#demo" : location.pathname);
  };
  const card = useScorecard(score);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "SELECT") return;
      if (e.key === "Escape") sendCmd({ cmd: "estop" });
      else if (e.key === " ") {
        e.preventDefault();
        sendCmd({ cmd: "grab" });
      } else if (e.key.toLowerCase() === "d") go(view === "demo" ? "console" : "demo"); else if (e.key === "Enter") sendCmd({ cmd: "release" });
      else if (e.key.toLowerCase() === "s") setScore((v) => !v);
      else if (/^[1-8]$/.test(e.key)) sendCmd({ cmd: "loops", k: Number(e.key) });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view]);

  if (view === "demo") return <DemoView tele={tele} link={link} onConsole={() => go("console")} />;

  return (
    <div className="app">
      <TopBar
        tele={tele}
        link={link}
        onStart={() => sendCmd({ cmd: "grab" })}
        onDemo={() => go("demo")}
        onStop={() => sendCmd({ cmd: "estop" })}
        onScore={() => setScore(true)}
      />
      <main className="main">
        <div className="col left">
          <CameraStage tele={tele} />
          <PhaseRail tele={tele} />
          <div className="split">
            <TablePlan tele={tele} />
            <LogFeed logs={logs} source={tele.source} />
          </div>
        </div>
        <div className="col right">
          <InferencePanel tele={tele} card={card} onLoops={(k) => sendCmd({ cmd: "loops", k })} />
          <Joints tele={tele} history={history} />
        </div>
      </main>
      {link === "down" && <div className="banner">bridge offline · run python -m demo_ui.bridge (--replay)</div>}
      {tele.phase === "estop" && (
        <div className="estop-frame">
          <span>STOPPED · press Run to resume</span>
        </div>
      )}
      {score && <Scorecard card={card} onClose={() => setScore(false)} />}
    </div>
  );
}
