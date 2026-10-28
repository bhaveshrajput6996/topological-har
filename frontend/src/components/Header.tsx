import { StatusDot } from "./ui";
import { useAppState } from "../state/AppState";
import { relTime } from "../lib/format";
import { useRoute } from "../state/router";

const TITLES: Record<string, { t: string; s: string }> = {
  overview: { t: "Overview", s: "End-to-end view of the HAR research pipeline" },
  live: { t: "Live Recognition", s: "Dataset replay through a trained model, streamed over WebSocket" },
  datasets: { t: "Dataset Explorer", s: "Datasets served by the backend adapters" },
  windows: { t: "Window Inspector", s: "Sensor window → point cloud → topology, per subject and index" },
  tda: { t: "TDA Analysis", s: "Vietoris-Rips persistent homology of the motion point cloud" },
  model: { t: "Model Analysis", s: "DeepConvLSTM architecture and TDA fusion" },
  evaluation: { t: "Evaluation", s: "Aggregate metrics, confusion matrix and resource profile" },
  experiments: { t: "Experiments", s: "Persistent experiment history from the SQLite store" },
  system: { t: "System / API Status", s: "Backend connectivity and request diagnostics" },
};

export function Header() {
  const route = useRoute();
  const { backendStatus, health, lastHealthCheck, selectedRun, selectedDataset, theme, toggleTheme } = useAppState();
  const info = TITLES[route.page] ?? { t: route.page, s: "" };
  const dot = backendStatus === "online" ? "ok" : backendStatus === "offline" ? "err" : "warn";

  return (
    <header className="header">
      <div style={{ minWidth: 0 }}>
        <div className="header-title">TOPOLOGICAL HAR</div>
        <div className="header-sub">Human Activity Recognition using Topological Data Analysis + DeepConvLSTM</div>
      </div>
      <div className="header-right">
        <div className="header-chip optional" title="Backend API health">
          <StatusDot status={dot} pulse={backendStatus === "online"} />
          <span className="val">
            API {backendStatus}
            {backendStatus === "online" && health ? ` · v${health.version}` : ""}
          </span>
          {lastHealthCheck && <span className="hint mono" style={{ fontSize: 9.5 }}>{relTime(lastHealthCheck)}</span>}
        </div>
        <div className="header-chip optional" title="Selected dataset">
          <span className="hint">DS</span>
          <span className="val" style={{ fontWeight: 600 }}>{selectedDataset ?? "—"}</span>
        </div>
        <div className="header-chip" title="Selected experiment run">
          <span className="hint">RUN</span>
          <span className="val mono">{selectedRun ? selectedRun.run_id : "—"}</span>
        </div>
        <button
          className="btn ghost sm"
          onClick={toggleTheme}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          style={{ padding: "6px 8px" }}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"><circle cx="12" cy="12" r="4.5" /><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M4.9 4.9l1.6 1.6M17.5 17.5l1.6 1.6M19.1 4.9l-1.6 1.6M6.5 17.5l-1.6 1.6" /></svg>
          ) : (
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d="M20.5 14.5A8.5 8.5 0 1 1 9.5 3.5a7 7 0 0 0 11 11Z" /></svg>
          )}
        </button>
        <div className="header-chip" title="Current page">
          <span className="val" style={{ color: "var(--accent)" }}>{info.t}</span>
        </div>
      </div>
    </header>
  );
}
