import type { CSSProperties, ReactNode } from "react";

/* ---------------- icons (inline SVG, stroke-based) ---------------- */
const S = { fill: "none", stroke: "currentColor", strokeWidth: 1.7, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

export const IconOverview = () => (
  <svg viewBox="0 0 24 24" {...S}><rect x="3" y="3" width="7" height="9" rx="1.5" /><rect x="14" y="3" width="7" height="5" rx="1.5" /><rect x="14" y="12" width="7" height="9" rx="1.5" /><rect x="3" y="16" width="7" height="5" rx="1.5" /></svg>
);
export const IconPulse = () => (
  <svg viewBox="0 0 24 24" {...S}><path d="M3 12h4l2-7 4 14 2-7h6" /></svg>
);
export const IconDatabase = () => (
  <svg viewBox="0 0 24 24" {...S}><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" /><path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" /></svg>
);
export const IconWindow = () => (
  <svg viewBox="0 0 24 24" {...S}><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M8 4v5" /></svg>
);
export const IconTopology = () => (
  <svg viewBox="0 0 24 24" {...S}><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="5" /><circle cx="12" cy="12" r="1.4" fill="currentColor" /></svg>
);
export const IconNetwork = () => (
  <svg viewBox="0 0 24 24" {...S}><circle cx="5" cy="12" r="2.2" /><circle cx="19" cy="5" r="2.2" /><circle cx="19" cy="19" r="2.2" /><path d="M7 11l9.5-5.2M7 13l9.5 5.2" /></svg>
);
export const IconChart = () => (
  <svg viewBox="0 0 24 24" {...S}><path d="M4 20V4" /><path d="M4 20h16" /><path d="M7 15l4-4 3 3 5-6" /></svg>
);
export const IconFlask = () => (
  <svg viewBox="0 0 24 24" {...S}><path d="M9 3h6M10 3v5.5L5.5 17A2.5 2.5 0 0 0 8 21h8a2.5 2.5 0 0 0 2.5-4L14 8.5V3" /><path d="M7 15h10" /></svg>
);
export const IconHistory = () => (
  <svg viewBox="0 0 24 24" {...S}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.5 2" /></svg>
);
export const IconServer = () => (
  <svg viewBox="0 0 24 24" {...S}><rect x="3" y="4" width="18" height="7" rx="2" /><rect x="3" y="13" width="18" height="7" rx="2" /><path d="M7 7.5h.01M7 16.5h.01" /></svg>
);

export const PageIcons = {
  overview: <IconOverview />,
  live: <IconPulse />,
  datasets: <IconDatabase />,
  windows: <IconWindow />,
  tda: <IconTopology />,
  model: <IconNetwork />,
  evaluation: <IconChart />,
  experiments: <IconFlask />,
  system: <IconServer />,
};

/* ---------------- primitives ---------------- */
export function Card({
  title, sub, tools, children, className, style, accent,
}: {
  title?: ReactNode;
  sub?: string;
  tools?: ReactNode;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  accent?: string;
}) {
  return (
    <section className={`card ${className ?? ""}`} style={style}>
      {accent && (
        <div style={{ position: "absolute", left: 0, top: 14, bottom: 14, width: 3, borderRadius: 3, background: accent }} />
      )}
      {(title || tools) && (
        <div className="card-head">
          {title ? <span className="card-title">{title}</span> : null}
          {tools && <div className="card-tools">{tools}</div>}
        </div>
      )}
      {sub && <p className="card-sub" style={{ marginTop: -4, marginBottom: 10 }}>{sub}</p>}
      {children}
    </section>
  );
}

export function StatCard({
  label, value, sub, accent, className, loading, sm, children,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  accent?: string;
  className?: string;
  loading?: boolean;
  sm?: boolean;
  children?: ReactNode;
}) {
  return (
    <div className={`stat ${className ?? ""}`} style={accent ? { borderColor: `${accent}44` } : undefined}>
      <div className="stat-label">
        {accent && <span className="dot" style={{ background: accent }} />}
        {label}
      </div>
      {loading ? (
        <div className="skeleton" style={{ height: 22, marginTop: 9, width: "70%" }} />
      ) : (
        <div className={`stat-value ${sm ? "sm" : ""}`} style={accent ? { color: accent } : undefined}>{value}</div>
      )}
      {sub && <div className="stat-sub">{sub}</div>}
      {children}
    </div>
  );
}

export function StatusDot({ status, pulse }: { status: "ok" | "err" | "warn" | "off"; pulse?: boolean }) {
  return <span className={`dot ${status} ${pulse ? "pulse" : ""}`} />;
}

export function Badge({ tone, children }: { tone?: "" | "green" | "red" | "amber" | "cyan" | "violet"; children: ReactNode }) {
  return <span className={`badge ${tone ?? ""}`}>{children}</span>;
}

export function Button({
  children, onClick, variant, size, disabled, title, type, style,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "" | "primary" | "danger" | "ghost";
  size?: "" | "sm";
  disabled?: boolean;
  title?: string;
  type?: "button" | "submit";
  style?: CSSProperties;
}) {
  return (
    <button
      type={type ?? "button"}
      className={`btn ${variant ?? ""} ${size ?? ""}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={style}
    >
      {children}
    </button>
  );
}

export function Field({
  label, children, style,
}: { label: string; children: ReactNode; style?: CSSProperties }) {
  return (
    <label className="field" style={style}>
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

export function Select({
  value, onChange, options, placeholder, disabled, style,
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
  disabled?: boolean;
  style?: CSSProperties;
}) {
  return (
    <select
      className="select"
      value={value}
      disabled={disabled}
      style={style}
      onChange={(e) => onChange(e.target.value)}
    >
      {placeholder !== undefined && <option value="">{placeholder}</option>}
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

/* ---------------- async states ---------------- */
export function Skeleton({ h = 200, w = "100%", style }: { h?: number | string; w?: number | string; style?: CSSProperties }) {
  return <div className="skeleton" style={{ height: h, width: w, ...style }} />;
}

export function LoadingBlock({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="state-block">
      <div style={{ display: "flex", gap: 4 }}>
        {[0, 1, 2].map((i) => (
          <span key={i} className="dot pulse" style={{ background: "var(--accent)", animationDelay: `${i * 0.18}s` }} />
        ))}
      </div>
      <span>{label}</span>
    </div>
  );
}

export function EmptyState({ message, hint, icon, actions }: { message: string; hint?: string; icon?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="state-block">
      {icon && <div className="icon">{icon}</div>}
      <div className="big">{message}</div>
      {hint && <div className="hint">{hint}</div>}
      {actions && <div className="actions">{actions}</div>}
    </div>
  );
}

export function ErrorState({
  error, retry, context,
}: {
  error: { message: string; status?: number } | null;
  retry?: () => void;
  context?: string;
}) {
  if (!error) return null;
  const offline = error.status === 0;
  return (
    <div className="state-block">
      <div className="error-banner" style={{ maxWidth: 560 }}>
        <span className="code">{offline ? "CONNECTION" : `HTTP ${error.status ?? "?"}`}</span>
        <span>
          {context ? <><b>{context}. </b></> : null}
          {offline
            ? "Backend is unreachable. Start it with: cd backend && .venv\\Scripts\\python -m uvicorn app.main:app"
            : error.message}
        </span>
      </div>
      {retry && (
        <div className="actions">
          <Button size="sm" onClick={retry}>Retry</Button>
        </div>
      )}
    </div>
  );
}

export function SectionNote({ tone, children }: { tone?: "" | "violet" | "green" | "amber"; children: ReactNode }) {
  return <div className={`note ${tone ?? ""}`}>{children}</div>;
}

export function KV({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v], i) => (
        <FragmentedKV key={i} k={k} v={v} />
      ))}
    </dl>
  );
}

function FragmentedKV({ k, v }: { k: string; v: ReactNode }) {
  return (
    <>
      <dt>{k}</dt>
      <dd>{v}</dd>
    </>
  );
}
