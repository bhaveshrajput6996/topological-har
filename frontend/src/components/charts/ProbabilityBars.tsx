import { useSize } from "./chartUtils";

/** Horizontal probability bars for the class-probability distribution. */
export function ProbabilityBars({
  probabilities,
  highlight,
  compact = false,
}: {
  probabilities: Record<string, number>;
  highlight?: string;
  compact?: boolean;
}) {
  const [ref, width] = useSize<HTMLDivElement>();
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1e-9, ...entries.map(([, v]) => v));
  const labelW = compact ? 110 : 150;
  const rowH = compact ? 17 : 21;

  if (entries.length === 0) return <div className="hint" style={{ padding: 10 }}>No probabilities</div>;

  return (
    <div ref={ref} style={{ width: "100%", display: "flex", flexDirection: "column", gap: 3 }}>
      {entries.map(([name, v]) => {
        const isTop = name === highlight;
        const color = isTop ? "var(--accent)" : "var(--violet)";
        return (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, height: rowH }}>
            <span
              className="mono"
              style={{
                width: labelW, fontSize: 10.5, color: isTop ? "var(--text)" : "var(--muted)",
                fontWeight: isTop ? 700 : 400, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                textAlign: "right", flexShrink: 0,
              }}
            >
              {name.replace(/_/g, " ")}
            </span>
            <div style={{ flex: 1, height: compact ? 8 : 11, background: "var(--panel-3)", borderRadius: 4, overflow: "hidden", position: "relative" }}>
              <div
                style={{
                  width: `${(v / max) * 100}%`, height: "100%",
                  background: color,
                  opacity: isTop ? 1 : 0.55,
                  borderRadius: 4,
                  transition: "width 0.25s ease",
                }}
              />
            </div>
            <span className="mono" style={{ width: 52, fontSize: 10.5, textAlign: "right", color: isTop ? "var(--text)" : "var(--muted)", flexShrink: 0 }}>
              {(v * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
      {width < 0 && null}
    </div>
  );
}
