import { useMemo, useState } from "react";
import { formatTick, linearScale, niceTicks, useSize } from "./chartUtils";

export interface ScatterPoint {
  x: number;
  y: number;
  color: string;
  size?: number;
  label?: string;
  sub?: string;
  opacity?: number;
}

/**
 * Generic 2-D scatter plot (persistence diagram, PCA point cloud).
 * Optional diagonal reference (death = birth) and axis labels.
 */
export function ScatterPlot({
  points,
  height = 260,
  xLabel,
  yLabel,
  diagonal = false,
  equalAspect = false,
  yTitle,
}: {
  points: ScatterPoint[];
  height?: number;
  xLabel?: string;
  yLabel?: string;
  diagonal?: boolean;
  equalAspect?: boolean;
  yTitle?: string;
}) {
  const [wrapRef, width] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ px: number; py: number; p: ScatterPoint } | null>(null);

  const m = { top: 14, right: 14, bottom: 28, left: 44 };
  const w = Math.max(width, 120);
  const iw = Math.max(10, w - m.left - m.right);
  const ih = Math.max(10, height - m.top - m.bottom);

  const { sx, sy, xTicks, yTicks } = useMemo(() => {
    if (points.length === 0) return { sx: null, sy: null, xTicks: [], yTicks: [] };
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    let x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (diagonal) {
      // persistence diagram: death >= birth, pad to square-ish view
      x0 = Math.min(0, x0); y0 = Math.min(0, y0);
      x1 = Math.max(x1, y1); y1 = Math.max(y1, x1);
      x0 = Math.max(0, x0 - (x1 - x0) * 0.06);
      y0 = Math.max(0, y0 - (y1 - y0) * 0.06);
      x1 = x1 + (x1 - x0) * 0.06;
      y1 = y1 + (y1 - y0) * 0.06;
      if (x0 === x1) { x1 = x0 + 1; y1 = y0 + 1; }
    } else {
      const px = (x1 - x0) * 0.07 || 0.5;
      const py = (y1 - y0) * 0.07 || 0.5;
      x0 -= px; x1 += px; y0 -= py; y1 += py;
    }
    const sX = linearScale([x0, x1], [0, iw]);
    const sY = linearScale([y0, y1], [ih, 0]);
    return {
      sx: sX,
      sy: sY,
      xTicks: niceTicks(x0, x1, 5),
      yTicks: niceTicks(y0, y1, 5),
    };
  }, [points, diagonal, iw, ih]);

  if (!sx || !sy || points.length === 0) {
    return (
      <div ref={wrapRef} style={{ width: "100%" }}>
        <div style={{ height, display: "grid", placeItems: "center" }} className="hint">No data</div>
      </div>
    );
  }

  // equal aspect: adjust so unit distances match (used by point cloud)
  const sxEq = sx;
  const syEq = sy;

  const diagonalD = diagonal
    ? `M${m.left + sx(Math.max(0, sx.invert(0)))},${m.top + sy(Math.max(0, sy.invert(0)))} L${m.left + iw},${m.top + ih}`
    : null;

  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <svg
        width={w}
        height={height}
        style={{ display: "block" }}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          setHover((h) => (h ? { ...h, px: e.clientX, py: e.clientY } : h));
        }}
      >
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={m.left + sxEq(t)} x2={m.left + sxEq(t)} y1={m.top} y2={m.top + ih} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={m.left + sxEq(t)} y={m.top + ih + 16} textAnchor="middle" fontSize="9.5" fill="var(--muted)" fontFamily="var(--mono)">{formatTick(t)}</text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.left} x2={m.left + iw} y1={m.top + syEq(t)} y2={m.top + syEq(t)} stroke="var(--border)" strokeDasharray="2 4" />
            <text x={m.left - 6} y={m.top + syEq(t) + 3} textAnchor="end" fontSize="9.5" fill="var(--muted)" fontFamily="var(--mono)">{formatTick(t)}</text>
          </g>
        ))}
        {diagonalD && (
          <line d={diagonalD} x1={m.left} y1={m.top + ih} x2={m.left + iw} y2={m.top} stroke="var(--border-strong)" strokeDasharray="4 4" />
        )}
        {diagonal && (
          <text x={m.left + iw - 4} y={m.top + ih - 6} textAnchor="end" fontSize="9.5" fill="var(--muted)" fontStyle="italic">
            death = birth
          </text>
        )}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={m.left + sxEq(p.x)}
            cy={m.top + syEq(p.y)}
            r={p.size ?? 3.4}
            fill={p.color}
            opacity={p.opacity ?? 0.85}
            stroke={hover?.p === p ? "var(--text)" : "var(--panel)"}
            strokeWidth={hover?.p === p ? 1.6 : 0.8}
            style={{ cursor: "crosshair" }}
            onMouseEnter={() => setHover({ px: 0, py: 0, p })}
          />
        ))}
        {xLabel && <text x={m.left + iw} y={height - 1} textAnchor="end" fontSize="9.5" fill="var(--muted)">{xLabel}</text>}
        {yTitle && <text x={4} y={m.top + 4} fontSize="9.5" fill="var(--muted)">{yTitle}</text>}
      </svg>
      {hover && (
        <div className="chart-tooltip" style={{ left: hover.px + 14, top: hover.py + 30 }}>
          {hover.p.label && <div style={{ fontWeight: 700 }}>{hover.p.label}</div>}
          {hover.p.sub && <div style={{ color: "var(--muted)" }}>{hover.p.sub}</div>}
          <div className="tt-row" style={{ marginTop: 2 }}>
            <span style={{ color: "var(--muted)" }}>{xLabel ?? "x"}</span>
            <span style={{ marginLeft: "auto" }}>{hover.p.x.toFixed(3)}</span>
          </div>
          <div className="tt-row">
            <span style={{ color: "var(--muted)" }}>{yLabel ?? "y"}</span>
            <span style={{ marginLeft: "auto" }}>{hover.p.y.toFixed(3)}</span>
          </div>
        </div>
      )}
    </div>
  );
}

/** sequential colormap for time-indexed point clouds */
export function timeColor(t: number, tMax: number): string {
  const u = tMax > 0 ? t / tMax : 0;
  const stops: [number, number, number][] = [
    [34, 100, 234],   // indigo-blue
    [34, 211, 238],   // cyan
    [52, 211, 153],   // green
    [251, 191, 36],   // amber
    [244, 114, 182],  // pink
  ];
  const x = Math.min(0.999, Math.max(0, u)) * (stops.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  const a = stops[i];
  const b = stops[Math.min(stops.length - 1, i + 1)];
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
