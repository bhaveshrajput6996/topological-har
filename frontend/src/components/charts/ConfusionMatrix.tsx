import { useMemo, useState } from "react";
import type { ConfusionRow } from "../../api/types";
import { useSize } from "./chartUtils";

/**
 * Confusion matrix heatmap (per-row normalized coloring = recall view),
 * built from the backend's sparse (true_label, predicted_label, count) rows.
 */
export function ConfusionMatrix({
  rows,
  cellSize = 44,
}: {
  rows: ConfusionRow[];
  cellSize?: number;
}) {
  const [hover, setHover] = useState<{ px: number; py: number; t: string; p: string; c: number; share: number } | null>(null);

  const { labels, grid, rowTotals } = useMemo(() => {
    const set = new Set<string>();
    for (const r of rows) { set.add(r.true_label); set.add(r.predicted_label); }
    const labels = [...set].sort();
    const idx = new Map(labels.map((l, i) => [l, i]));
    const n = labels.length;
    const grid = Array.from({ length: n }, () => Array.from({ length: n }, () => 0));
    const rowTotals = Array.from({ length: n }, () => 0);
    for (const r of rows) {
      const i = idx.get(r.true_label)!;
      const j = idx.get(r.predicted_label)!;
      grid[i][j] += r.count;
      rowTotals[i] += r.count;
    }
    return { labels, grid, rowTotals };
  }, [rows]);

  if (labels.length === 0) {
    return <div className="hint" style={{ padding: 14 }}>No confusion matrix stored for this run</div>;
  }

  const n = labels.length;
  const labelW = Math.min(150, Math.max(90, Math.max(...labels.map((l) => l.length)) * 7.4));
  const w = labelW + n * cellSize + 8;
  const h = n * cellSize + 58;

  const cellColor = (count: number, rowTotal: number, correct: boolean): string => {
    if (!count) return "var(--panel-2)";
    const u = rowTotal > 0 ? count / rowTotal : 0;
    // green diagonal (correct), red off-diagonal (errors)
    const r = correct ? Math.round(120 - 70 * u) : Math.round(252 - 200 * u);
    const g = correct ? Math.round(190 + 60 * u) : Math.round(80 + 80 * u);
    const b = correct ? Math.round(140 - 30 * u) : Math.round(90 - 30 * u);
    return `rgba(${r},${g},${b},${0.16 + 0.74 * u})`;
  };

  return (
    <div className="scroll-x">
      <svg width={w} height={h} style={{ display: "block", minWidth: w }}>
        {grid.map((row, i) =>
          row.map((count, j) => {
            const share = rowTotals[i] > 0 ? count / rowTotals[i] : 0;
            const isHover = hover && hover.t === labels[i] && hover.p === labels[j];
            return (
              <g key={`${i}-${j}`}>
                <rect
                  x={labelW + j * cellSize}
                  y={8 + i * cellSize}
                  width={cellSize - 3}
                  height={cellSize - 3}
                  rx={4}
                  fill={cellColor(count, rowTotals[i], i === j)}
                  stroke={isHover ? "var(--text)" : "var(--border)"}
                  strokeWidth={isHover ? 1.6 : 1}
                  onMouseEnter={() =>
                    setHover({ px: 0, py: 0, t: labels[i], p: labels[j], c: count, share })
                  }
                  onMouseMove={(e) =>
                    setHover({ px: e.clientX, py: e.clientY, t: labels[i], p: labels[j], c: count, share })
                  }
                  onMouseLeave={() => setHover(null)}
                />
                {count > 0 && (
                  <text
                    x={labelW + j * cellSize + (cellSize - 3) / 2}
                    y={8 + i * cellSize + (cellSize - 3) / 2 + 3.5}
                    textAnchor="middle"
                    fontSize={count > 999 ? 9 : 11}
                    fontFamily="var(--mono)"
                    fill={share > 0.55 ? "#fff" : "var(--text-dim)"}
                  >
                    {count}
                  </text>
                )}
              </g>
            );
          }),
        )}
        {/* row labels (true) */}
        {labels.map((l, i) => (
          <text
            key={`rl${i}`}
            x={labelW - 8}
            y={8 + i * cellSize + cellSize / 2}
            textAnchor="end"
            fontSize="10"
            fill="var(--text-dim)"
            fontFamily="var(--mono)"
            dominantBaseline="middle"
          >
            {l.replace(/_/g, " ")}
          </text>
        ))}
        {/* col labels (predicted) */}
        {labels.map((l, j) => (
          <text
            key={`cl${j}`}
            x={labelW + j * cellSize + (cellSize - 3) / 2}
            y={h - 44}
            textAnchor="start"
            fontSize="9.5"
            fill="var(--muted)"
            fontFamily="var(--mono)"
            transform={`rotate(58 ${labelW + j * cellSize + (cellSize - 3) / 2} ${h - 44})`}
          >
            {l.replace(/_/g, " ")}
          </text>
        ))}
      </svg>
      {hover && (
        <div className="chart-tooltip" style={{ left: hover.px + 14, top: hover.py + 26 }}>
          <div className="tt-row"><span style={{ color: "var(--muted)" }}>true</span><span style={{ marginLeft: "auto" }}>{hover.t}</span></div>
          <div className="tt-row"><span style={{ color: "var(--muted)" }}>pred</span><span style={{ marginLeft: "auto" }}>{hover.p}</span></div>
          <div className="tt-row" style={{ borderTop: "1px solid var(--border)", marginTop: 3, paddingTop: 3 }}>
            <span style={{ color: "var(--muted)" }}>windows</span><span style={{ marginLeft: "auto" }}>{hover.c}</span>
          </div>
          <div className="tt-row"><span style={{ color: "var(--muted)" }}>row share</span><span style={{ marginLeft: "auto" }}>{(hover.share * 100).toFixed(1)}%</span></div>
        </div>
      )}
    </div>
  );
}
