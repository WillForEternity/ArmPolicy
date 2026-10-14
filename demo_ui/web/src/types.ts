export type Phase = "idle" | "pick" | "lift" | "hold" | "release" | "home" | "estop";
export type XY = [number, number];

export interface Box {
  xy?: XY; // base frame, metres (CONTRACT §4: forward = −y, +x = robot's left)
  theta_deg?: number; // long side, CCW from +x
  corners_px?: XY[]; // normalised 0–1 in the camera frame
  latched?: boolean; // pose held from arm-clear frames
  size_m?: XY;
}

export interface Infer {
  host?: "uno-q" | "mac" | null;
  ms?: number;
  ms_hist?: number[];
  loops?: number;
  model?: string;
  hz?: number;
  available_loops?: number[];
}

export interface Telemetry {
  t?: number;
  mode?: "live" | "replay";
  source?: string;
  phase?: Phase;
  phase_t?: number;
  episode?: number;
  state?: number[];
  action?: number[];
  chunk?: number[][];
  box?: Box;
  infer?: Infer;
  board?: { cpu_pct?: number; ram_mb?: number; temp_c?: number; power_w?: number };
  links?: { camera?: boolean; follower?: boolean; board?: boolean };
  grasp_err?: number;
  grasped?: boolean;
  pan_bearing_sign?: number;
  video?: boolean; // the bridge is streaming camera frames at /video.mjpg
  task?: "pick" | "record"; // record: teleop data collection (grab = start, release = save, estop = discard)
  saved?: number; // episodes saved this session (record)
  calib?: { workspace_px?: XY[]; crosses_px?: XY[] };
}

export interface LogEntry {
  t: number;
  level: string;
  text: string;
}

export interface ScoreRow {
  metric: string;
  unit: string;
  baseline: number | null;
  optimized: number | null;
  better: "lower" | "higher";
  on: string;
  note?: string;
}

export interface Scorecard {
  baseline: { name: string; detail: string };
  optimized: { name: string; detail: string };
  rows: ScoreRow[];
  loop_curve?: { source: string; points: Record<string, number> };
}
