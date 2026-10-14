export const JOINTS = [
  { key: "shoulder_pan", label: "Pan", range: [-110, 110] },
  { key: "shoulder_lift", label: "Lift", range: [-110, 110] },
  { key: "elbow_flex", label: "Elbow", range: [-110, 110] },
  { key: "wrist_flex", label: "Wrist", range: [-110, 110] },
  { key: "wrist_roll", label: "Roll", range: [-160, 160] },
  { key: "gripper", label: "Grip", range: [0, 100] },
] as const;

// CONTRACT §4 workspace and calibration crosses, metres in the base frame.
export const WORKSPACE = { rMin: 0.12, rMax: 0.28, bearing: 34 };
export const CROSSES: { id: string; x: number; y: number }[] = [
  { id: "A", x: 0.1, y: -0.15 },
  { id: "B", x: -0.1, y: -0.15 },
  { id: "C", x: 0.1, y: -0.3 },
  { id: "D", x: -0.1, y: -0.3 },
];
export const BOX_SIZE: [number, number] = [0.089, 0.066];
export const CONTROL_HZ = 30;

export const PHASES = [
  { id: "pick", label: "Pick", who: "learned", weight: 5 },
  { id: "lift", label: "Lift", who: "scripted", weight: 2 },
  { id: "hold", label: "Hold", who: "scripted", weight: 1.4 },
  { id: "release", label: "Release", who: "scripted", weight: 1.1 },
  { id: "home", label: "Home", who: "scripted", weight: 2.2 },
] as const;
