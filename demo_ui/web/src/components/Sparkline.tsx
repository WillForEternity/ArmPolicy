interface Props {
  values: number[];
  w: number;
  h: number;
  min?: number;
  max?: number;
  stroke?: string;
  guide?: number; // horizontal reference line in value units
}

export default function Sparkline({ values, w, h, min, max, stroke = "var(--dim)", guide }: Props) {
  if (values.length < 2) return <svg width={w} height={h} />;
  const lo = min ?? Math.min(...values);
  const hi = max ?? Math.max(...values);
  const span = hi - lo || 1;
  const y = (v: number) => h - 1 - ((v - lo) / span) * (h - 2);
  const step = w / (values.length - 1);
  const d = values.map((v, i) => `${i ? "L" : "M"}${(i * step).toFixed(1)},${y(v).toFixed(1)}`).join("");
  return (
    <svg width={w} height={h} style={{ display: "block", overflow: "visible" }}>
      {guide !== undefined && guide >= lo && guide <= hi && (
        <line x1={0} x2={w} y1={y(guide)} y2={y(guide)} stroke="var(--line-2)" strokeDasharray="2 3" />
      )}
      <path d={d} fill="none" stroke={stroke} strokeWidth={1.25} strokeLinejoin="round" />
      <circle cx={w} cy={y(values[values.length - 1])} r={2} fill={stroke} />
    </svg>
  );
}
