import { linearScale } from "./chartUtils";

/** Tiny inline sparkline (used on stat cards). */
export function Sparkline({
  values,
  width = 64,
  height = 22,
  color = "var(--accent)",
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
}) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const sy = linearScale([min - (max - min || 1) * 0.1, max + (max - min || 1) * 0.1], [height - 2, 2]);
  const sx = linearScale([0, values.length - 1], [1, width - 1]);
  const d = values.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
  return (
    <svg className="spark" width={width} height={height} style={{ pointerEvents: "none" }}>
      <path d={d} fill="none" stroke={color} strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} />
    </svg>
  );
}
