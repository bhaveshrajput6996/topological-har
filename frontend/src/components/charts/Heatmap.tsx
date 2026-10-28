import { useState } from "react";
import { useSize } from "./chartUtils";

/**
 * Grid heatmap for the persistence image (values normalized [0,1]).
 * Rendered with the origin at bottom-left: x = birth, y = persistence.
 */
export function Heatmap({
  values,
  rows,
  cols,
  height = 260,
  xLabel = "birth (normalized)",
  yLabel = "persistence (normalized)",
  colormap = "topo",
}: {
  values: number[];
  rows: number;
  cols: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  colormap?: "topo" | "viridis";
}) {
  const [wrapRef, width] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ px: number; py: number; r: number; c: number; v: number } | null>(null);

  const m = { top: 10, right: 12, bottom: 30, left: 40 };
  const w = Math.max(width, 120);
  const iw = Math.max(10, w - m.left - m.right);
  const ih = Math.max(10, height - m.top - m.bottom);

  if (!values || values.length !== rows * cols) {
    return (
      <div ref={wrapRef} style={{ width: "100%" }}>
        <div style={{ height, display: "grid", placeItems: "center" }} className="hint">No image data</div>
      </div>
    );
  }

  const max = Math.max(...values);
  if (max <= 0) {
    return (
      <div ref={wrapRef} style={{ width: "100%" }}>
        <div style={{ height, display: "grid", placeItems: "center" }} className="hint">All-zero image</div>
      </div>
    );
  }

  const cw = iw / cols;
  const chh = ih / rows;

  const cellColor = (v: number): string => {
    const u = Math.pow(Math.min(1, v / max), 0.6); // gamma for visibility
    if (colormap === "viridis") {
      const r = Math.round(68 + (240 - 68) * u);
      const g = Math.round(1 + (249 - 1) * u);
      const b = Math.round(84 + (33 - 84) * u);
      return `rgb(${r},${g},${b})`;
    }
    // cyan -> violet -> warm research ramp
    if (u < 0.5) {
      const f = u / 0.5;
      return `rgba(${Math.round(8 + 150 * f)},${Math.round(110 + 60 * f)},${Math.round(140 + 80 * f)},${0.25 + 0.7 * f})`;
    }
    const f = (u - 0.5) / 0.5;
    return `rgba(${Math.round(158 + 90 * f)},${Math.round(170 - 20 * f)},${Math.round(220 - 30 * f)},${0.95})`;
  };

  const rectFor = (r: number, c: number) => ({
    x: m.left + c * cw,
    y: m.top + (rows - 1 - r) * chh,
  });

  return (
    <div ref={wrapRef} style={{ width: "100%" }}>
      <svg width={w} height={height} style={{ display: "block" }} onMouseLeave={() => setHover(null)}>
        {Array.from({ length: rows }, (_, r) =>
          Array.from({ length: cols }, (_, c) => {
            const v = values[r * cols + c];
            const rect = rectFor(r, c);
            const isHover = hover && hover.r === r && hover.c === c;
            return (
              <rect
                key={`${r}-${c}`}
                x={rect.x}
                y={rect.y}
                width={cw + 0.5}
                height={chh + 0.5}
                fill={cellColor(v)}
                stroke={isHover ? "var(--text)" : "none"}
                strokeWidth={isHover ? 1.4 : 0}
                onMouseEnter={() =>
                  setHover({ px: 0, py: 0, r, c, v })
                }
                onMouseMove={(e) => setHover({ px: e.clientX, py: e.clientY, r, c, v })}
              />
            );
          }),
        )}
        {/* axes */}
        <line x1={m.left} y1={m.top + ih} x2={m.left + iw} y2={m.top + ih} stroke="var(--border-strong)" />
        <line x1={m.left} y1={m.top} x2={m.left} y2={m.top + ih} stroke="var(--border-strong)" />
        <text x={m.left + iw} y={height - 4} textAnchor="end" fontSize="9.5" fill="var(--muted)">{xLabel}</text>
        <text x={6} y={m.top + 2} fontSize="9.5" fill="var(--muted)">{yLabel}</text>
        {/* colorbar */}
        <defs>
          <linearGradient id="hm-cb" x1="0" y1="1" x2="0" y2="0">
            {Array.from({ length: 8 }, (_, i) => (
              <stop key={i} offset={i / 7} stopColor={cellColor(max * Math.pow(i / 7, 1 / 0.6))} />
            ))}
          </linearGradient>
        </defs>
        <rect x={w - 14} y={m.top} width={8} height={ih * 0.5} fill="url(#hm-cb)" rx={2} stroke="var(--border)" />
        <text x={w - 16} y={m.top + ih * 0.5 + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="var(--mono)">
          {max.toFixed(2)}
        </text>
      </svg>
      {hover && (
        <div className="chart-tooltip" style={{ left: hover.px + 14, top: hover.py + 30 }}>
          <div className="tt-row"><span style={{ color: "var(--muted)" }}>cell</span><span style={{ marginLeft: "auto" }}>[{hover.c}, {hover.r}]</span></div>
          <div className="tt-row"><span style={{ color: "var(--muted)" }}>value</span><span style={{ marginLeft: "auto" }}>{hover.v.toFixed(4)}</span></div>
        </div>
      )}
    </div>
  );
}
