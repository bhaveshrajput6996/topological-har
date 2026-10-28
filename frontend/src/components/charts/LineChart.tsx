import { useState } from "react";
import { formatTick, linearScale, niceTicks, useSize } from "./chartUtils";

export interface LineSeries {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
}

/**
 * Multi-series line chart with grid, axes, crosshair tooltip.
 * Pure SVG; resizes with the container.
 */
export function LineChart({
  series,
  height = 220,
  yLabel,
  xLabel,
  yDomain,
  area = false,
  minY,
  maxY,
}: {
  series: LineSeries[];
  height?: number;
  yLabel?: string;
  xLabel?: string;
  yDomain?: [number, number];
  area?: boolean;
  minY?: number;
  maxY?: number;
}) {
  const [wrapRef, width] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ x: number; data: { series: string; color: string; x: number; y: number }[] } | null>(null);
  const [cursor, setCursor] = useState<{ cx: number; cy: number } | null>(null);

  const m = { top: 12, right: 14, bottom: 26, left: 46 };
  const w = Math.max(width, 120);
  const iw = Math.max(10, w - m.left - m.right);
  const ih = Math.max(10, height - m.top - m.bottom);

  const allPts = series.flatMap((s) => s.points);
  if (allPts.length === 0) {
    return (
      <div ref={wrapRef} style={{ width: "100%" }}>
        <div style={{ height, display: "grid", placeItems: "center" }} className="hint">
          No data
        </div>
      </div>
    );
  }
  const xs = allPts.map((p) => p.x);
  const ys = allPts.map((p) => p.y);
  let y0 = yDomain ? yDomain[0] : (minY !== undefined ? Math.min(minY, Math.min(...ys)) : Math.min(...ys));
  let y1 = yDomain ? yDomain[1] : (maxY !== undefined ? Math.max(maxY, Math.max(...ys)) : Math.max(...ys));
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const padY = (y1 - y0) * 0.06;
  const sx = linearScale([Math.min(...xs), Math.max(...xs)], [0, iw]);
  const sy = linearScale([y0 - padY, y1 + padY], [ih, 0]);

  const yTicks = niceTicks(y0 - padY, y1 + padY, 5);
  const xTicks = niceTicks(Math.min(...xs), Math.max(...xs), Math.min(7, Math.max(2, iw / 90)));

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left - m.left;
    if (px < -8 || px > iw + 8) { setHover(null); setCursor(null); return; }
    const xVal = sx.invert(Math.max(0, Math.min(iw, px)));
    const data: { series: string; color: string; x: number; y: number }[] = [];
    for (const s of series) {
      if (s.points.length === 0) continue;
      let best = s.points[0];
      let bd = Math.abs(s.points[0].x - xVal);
      for (const p of s.points) {
        const d = Math.abs(p.x - xVal);
        if (d < bd) { bd = d; best = p; }
      }
      data.push({ series: s.name, color: s.color, x: best.x, y: best.y });
    }
    setHover({ x: e.clientX, data });
    setCursor({ cx: sx(xVal), cy: e.clientY - rect.top });
  };

  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <svg
        width={w}
        height={height}
        onMouseMove={onMove}
        onMouseLeave={() => { setHover(null); setCursor(null); }}
        style={{ display: "block" }}
      >
        {/* y grid */}
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={m.left} x2={m.left + iw} y1={m.top + sy(t)} y2={m.top + sy(t)} stroke="var(--border)" strokeDasharray="3 4" />
            <text x={m.left - 7} y={m.top + sy(t) + 3.5} textAnchor="end" fontSize="10" fill="var(--muted)" fontFamily="var(--mono)">
              {formatTick(t)}
            </text>
          </g>
        ))}
        {/* x grid */}
        {xTicks.map((t) => (
          <g key={`x${t}`}>
            <line x1={m.left + sx(t)} x2={m.left + sx(t)} y1={m.top} y2={m.top + ih} stroke="var(--border)" strokeDasharray="3 4" />
            <text x={m.left + sx(t)} y={m.top + ih + 15} textAnchor="middle" fontSize="10" fill="var(--muted)" fontFamily="var(--mono)">
              {formatTick(t)}
            </text>
          </g>
        ))}
        {yLabel && (
          <text x={12} y={m.top + 2} fontSize="10" fill="var(--muted)" fontFamily="var(--mono)">{yLabel}</text>
        )}
        {xLabel && (
          <text x={m.left + iw} y={height - 2} textAnchor="end" fontSize="10" fill="var(--muted)" fontFamily="var(--mono)">{xLabel}</text>
        )}
        {/* series */}
        {series.map((s) => {
          const d = s.points
            .map((p, i) => `${i === 0 ? "M" : "L"}${(m.left + sx(p.x)).toFixed(2)},${(m.top + sy(p.y)).toFixed(2)}`)
            .join(" ");
          const areaPath = area && s.points.length > 1
            ? `${d} L${(m.left + sx(s.points[s.points.length - 1].x)).toFixed(2)},${m.top + ih} L${(m.left + sx(s.points[0].x)).toFixed(2)},${m.top + ih} Z`
            : null;
          return (
            <g key={s.name}>
              {areaPath && <path d={areaPath} fill={s.color} opacity={0.09} />}
              <path
                d={d}
                fill="none"
                stroke={s.color}
                strokeWidth={1.8}
                strokeDasharray={s.dashed ? "5 4" : undefined}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </g>
          );
        })}
        {/* crosshair */}
        {cursor && hover && (
          <g>
            <line x1={m.left + cursor.cx} x2={m.left + cursor.cx} y1={m.top} y2={m.top + ih} stroke="var(--border-strong)" />
            {hover.data.map((d, i) => (
              <circle key={i} cx={m.left + sx(d.x)} cy={m.top + sy(d.y)} r={3} fill={d.color} stroke="var(--panel)" strokeWidth={1.5} />
            ))}
          </g>
        )}
      </svg>
      {hover && (
        <div className="chart-tooltip" style={{ left: hover.x + 14, top: (cursor?.cy ?? 0) + 40 }}>
          {hover.data.map((d, i) => (
            <div key={i} className="tt-row">
              <span className="swatch" style={{ background: d.color }} />
              <span style={{ color: "var(--muted)" }}>{d.series}</span>
              <span style={{ marginLeft: "auto" }}>{d.y.toFixed(3)}</span>
            </div>
          ))}
          <div className="tt-row" style={{ borderTop: "1px solid var(--border)", marginTop: 3, paddingTop: 3 }}>
            <span style={{ color: "var(--muted)" }}>{xLabel ?? "x"}</span>
            <span style={{ marginLeft: "auto" }}>{hover.data[0]?.x.toFixed(0)}</span>
          </div>
        </div>
      )}
    </div>
  );
}
