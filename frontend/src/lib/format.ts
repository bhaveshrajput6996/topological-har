export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(digits);
}

export function fmtMs(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  return `${v.toFixed(digits)} ms`;
}

export function fmtS(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v) || v < 0) return "N/A";
  if (v >= 60) return `${Math.floor(v / 60)}m ${Math.round(v % 60)}s`;
  return `${v.toFixed(1)} s`;
}

export function fmtMb(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v) || v < 0) return "N/A";
  return `${v.toFixed(1)} MB`;
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "N/A";
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      year: "numeric", month: "short", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return iso;
  }
}

export function relTime(ts: number): string {
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 5) return "just now";
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.round(m / 60)}h ago`;
}

/** The experiment summary uses -1 sentinels from SQL COALESCE for missing metrics. */
export function metricOrNull(v: number | null | undefined): number | null {
  if (v === null || v === undefined || Number.isNaN(v) || v < 0) return null;
  return v;
}

export function prettyLabel(s: string | null | undefined): string {
  if (!s) return "N/A";
  return s.replace(/_/g, " ");
}

export function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`;
}
