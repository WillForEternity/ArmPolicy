import { useCallback, useEffect, useRef, useState } from "react";
import type { LogEntry, Scorecard, Telemetry } from "./types";

const HISTORY = 120; // frames of joint history kept for sparklines (4 s at 30 Hz)

export type Link = "connecting" | "open" | "stale" | "down";

/** Bridge telemetry over SSE, coalesced to at most one React update per ~16 ms. */
export function useTelemetry() {
  const [tele, setTele] = useState<Telemetry>({});
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [link, setLink] = useState<Link>("connecting");
  const history = useRef<number[][]>([]);
  const latest = useRef<Telemetry>({});
  const lastAt = useRef(0);
  const raf = useRef(0);

  useEffect(() => {
    const es = new EventSource("/events");
    es.onopen = () => setLink("open");
    es.onerror = () => setLink("down");
    es.addEventListener("state", (e) => {
      const t = JSON.parse((e as MessageEvent).data) as Telemetry;
      latest.current = t;
      lastAt.current = performance.now();
      if (t.state) {
        history.current.push(t.state);
        if (history.current.length > HISTORY) history.current.shift();
      }
      if (!raf.current)
        raf.current = window.setTimeout(() => {
          raf.current = 0;
          setTele(latest.current);
        }, 16);
    });
    es.addEventListener("log", (e) => {
      const entry = JSON.parse((e as MessageEvent).data) as LogEntry;
      setLogs((l) => [...l.slice(-60), entry]);
    });
    const watchdog = setInterval(() => {
      if (es.readyState !== EventSource.OPEN) return;
      setLink(performance.now() - lastAt.current > 1500 ? "stale" : "open");
    }, 500);
    return () => {
      es.close();
      clearInterval(watchdog);
      clearTimeout(raf.current);
      raf.current = 0;
    };
  }, []);

  return { tele, logs, link, history: history.current };
}

export function sendCmd(cmd: Record<string, unknown>) {
  return fetch("/cmd", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cmd) }).catch(
    () => undefined,
  );
}

export function useScorecard(open: boolean) {
  const [card, setCard] = useState<Scorecard | null>(null);
  useEffect(() => {
    fetch("/scorecard.json")
      .then((r) => r.json())
      .then((j) => (j && j.rows ? setCard(j) : null))
      .catch(() => undefined);
  }, [open]);
  return card;
}

const store = {
  get: (k: string) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  set: (k: string, v: string) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* private mode */
    }
  },
};

/** Browser camera with device choice. With no saved choice it waits for the bridge: if a runtime
 * already owns the camera and streams frames (tele.video), show those instead of opening the
 * iPhone a second time; otherwise prefer the iPhone (Continuity Camera). */
export function useCamera(tele: { mode?: string; video?: boolean }) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string | null>(store.get("cam"));
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const all = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    setDevices(all);
    return all;
  }, []);

  useEffect(() => {
    if (!navigator.mediaDevices) {
      setError("camera API unavailable (needs localhost or https)");
      return;
    }
    let cancelled = false;
    let active: MediaStream | null = null;
    (async () => {
      try {
        let id = deviceId;
        if (id === "none") return setStream(null);
        if (id === "bridge") return setStream(null);
        if (!id) {
          if (!tele.mode) return; // no telemetry yet; decide once the bridge has spoken
          if (tele.video) return setDeviceId("bridge"); // not saved: only while a runtime owns the camera
          // a first, permissionless call returns no labels; ask once, then pick the iPhone
          const probe = await navigator.mediaDevices.getUserMedia({ video: true });
          probe.getTracks().forEach((t) => t.stop());
          const all = await refresh();
          id = (all.find((d) => /iphone|continuity/i.test(d.label)) ?? all[0])?.deviceId ?? null;
          if (!id) return setError("no camera found");
          setDeviceId(id);
          return;
        }
        active = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: id }, width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
        if (cancelled) return active.getTracks().forEach((t) => t.stop());
        setStream(active);
        setError(null);
        refresh();
      } catch (e) {
        setError((e as Error).message || "camera failed");
      }
    })();
    return () => {
      cancelled = true;
      active?.getTracks().forEach((t) => t.stop());
    };
  }, [deviceId, refresh, tele.mode, tele.video]);

  const choose = (id: string) => {
    store.set("cam", id);
    setDeviceId(id);
  };
  return { devices, deviceId, choose, stream, error };
}

/** Wall-clock that re-renders once a second. */
export function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}
