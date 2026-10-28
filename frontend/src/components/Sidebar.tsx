import { PageIcons, StatusDot } from "./ui";
import { navigate, useRoute } from "../state/router";
import { useAppState } from "../state/AppState";

const NAV: { section?: string; items: { id: string; label: string }[] }[] = [
  {
    section: "Research",
    items: [
      { id: "overview", label: "Overview" },
      { id: "live", label: "Live Recognition" },
    ],
  },
  {
    section: "Data & Topology",
    items: [
      { id: "datasets", label: "Dataset Explorer" },
      { id: "windows", label: "Window Inspector" },
      { id: "tda", label: "TDA Analysis" },
    ],
  },
  {
    section: "Models & Results",
    items: [
      { id: "model", label: "Model Analysis" },
      { id: "evaluation", label: "Evaluation" },
      { id: "experiments", label: "Experiments" },
    ],
  },
  {
    section: "Platform",
    items: [
      { id: "system", label: "System / API Status" },
    ],
  },
];

export function Sidebar() {
  const route = useRoute();
  const { backendStatus, selectedRun, selectedDataset } = useAppState();
  const dot = backendStatus === "online" ? "ok" : backendStatus === "offline" ? "err" : "warn";

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="9.5" stroke="var(--cyan)" strokeWidth="1.8" />
            <circle cx="12" cy="12" r="5" stroke="var(--violet)" strokeWidth="1.8" />
            <circle cx="12" cy="12" r="1.6" fill="var(--cyan)" />
          </svg>
        </div>
        <div>
          <div className="brand-name">TOPOLOGICAL HAR</div>
          <div className="brand-sub">TDA · DeepConvLSTM</div>
        </div>
      </div>
      <nav className="nav">
        {NAV.map((group) => (
          <div key={group.section ?? "plain"}>
            {group.section && <div className="nav-section">{group.section}</div>}
            {group.items.map((item) => (
              <div
                key={item.id}
                className={`nav-item ${route.page === item.id ? "active" : ""}`}
                onClick={() => navigate(item.id)}
              >
                <span className="nav-icon">{PageIcons[item.id as keyof typeof PageIcons]}</span>
                <span>{item.label}</span>
              </div>
            ))}
          </div>
        ))}
      </nav>
      <div className="sidebar-foot">
        <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
          <StatusDot status={dot} pulse={backendStatus === "online"} />
          <span>
            {backendStatus === "online" ? "Backend online" : backendStatus === "offline" ? "Backend offline" : "Checking…"}
          </span>
        </div>
        <div className="mono" style={{ fontSize: 10 }}>
          run: {selectedRun ? selectedRun.run_id : "—"} · ds: {selectedDataset ?? "—"}
        </div>
      </div>
    </aside>
  );
}
