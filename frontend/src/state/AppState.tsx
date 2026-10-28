import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { ExperimentsResponse, ExperimentRow, HealthResponse } from "../api/types";
import { useInterval } from "../api/hooks";

export type BackendStatus = "checking" | "online" | "offline";

interface AppState {
  theme: "dark" | "light";
  toggleTheme: () => void;
  backendStatus: BackendStatus;
  health: HealthResponse | null;
  lastHealthCheck: number | null;
  checkBackend: () => Promise<void>;
  experiments: ExperimentRow[];
  experimentsLoading: boolean;
  refreshExperiments: () => Promise<void>;
  selectedRunId: string | null;
  setSelectedRunId: (id: string | null) => void;
  selectedDataset: string | null;
  setSelectedDataset: (name: string | null) => void;
  selectedRun: ExperimentRow | null;
}

const Ctx = createContext<AppState | null>(null);

function persisted(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<"dark" | "light">(
    () => (persisted("thar.theme") as "dark" | "light" | null) ?? "dark",
  );
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [lastHealthCheck, setLastHealthCheck] = useState<number | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRow[]>([]);
  const [experimentsLoading, setExperimentsLoading] = useState(true);
  const [selectedRunId, setSelectedRunIdRaw] = useState<string | null>(() =>
    persisted("thar.runId"),
  );
  const [selectedDataset, setSelectedDatasetRaw] = useState<string | null>(() =>
    persisted("thar.dataset"),
  );

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("thar.theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);

  const checkBackend = useCallback(async () => {
    try {
      const h = await api.get<HealthResponse>("/health", 6000);
      setHealth(h);
      setBackendStatus(h.status === "ok" ? "online" : "offline");
    } catch {
      setHealth(null);
      setBackendStatus("offline");
    } finally {
      setLastHealthCheck(Date.now());
    }
  }, []);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  useInterval(() => {
    void checkBackend();
  }, backendStatus === "offline" ? 8000 : 15000);

  const refreshExperiments = useCallback(async () => {
    setExperimentsLoading(true);
    try {
      const res = await api.get<ExperimentsResponse>("/experiments", 20000);
      setExperiments(res.experiments);
      setSelectedRunIdRaw((cur) => {
        if (cur && res.experiments.some((e) => e.run_id === cur)) return cur;
        // auto-select the first completed run with metrics (prefer TDA fusion runs)
        const fusion = res.experiments.find(
          (e) => e.accuracy > 0 && e.model.includes("tda"),
        );
        const first = res.experiments.find((e) => e.accuracy > 0);
        const pick = fusion ?? first ?? null;
        try {
          if (pick) localStorage.setItem("thar.runId", pick.run_id);
          else localStorage.removeItem("thar.runId");
        } catch {
          /* ignore */
        }
        return pick?.run_id ?? null;
      });
    } catch {
      setExperiments([]);
    } finally {
      setExperimentsLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshExperiments();
  }, [refreshExperiments]);

  const setSelectedRunId = useCallback((id: string | null) => {
    setSelectedRunIdRaw(id);
    try {
      if (id) localStorage.setItem("thar.runId", id);
      else localStorage.removeItem("thar.runId");
    } catch {
      /* ignore */
    }
  }, []);

  const setSelectedDataset = useCallback((name: string | null) => {
    setSelectedDatasetRaw(name);
    try {
      if (name) localStorage.setItem("thar.dataset", name);
      else localStorage.removeItem("thar.dataset");
    } catch {
      /* ignore */
    }
  }, []);

  const selectedRun = useMemo(
    () => experiments.find((e) => e.run_id === selectedRunId) ?? null,
    [experiments, selectedRunId],
  );

  const value: AppState = {
    theme,
    toggleTheme: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
    backendStatus,
    health,
    lastHealthCheck,
    checkBackend,
    experiments,
    experimentsLoading,
    refreshExperiments,
    selectedRunId,
    setSelectedRunId,
    selectedDataset,
    setSelectedDataset,
    selectedRun,
  };
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAppState(): AppState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAppState must be used inside AppProvider");
  return ctx;
}
