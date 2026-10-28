import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useApi } from "../api/hooks";
import type { DatasetSubjectsResponse, DatasetsListResponse, InspectResponse } from "../api/types";
import { useAppState } from "../state/AppState";

/**
 * Shared dataset→subject→window selection state + /windows/inspect fetch.
 * Used by both the Window Inspector and the TDA Analysis pages.
 */
export function useInspection(initialWindowIndex = 0) {
  const { experiments, selectedRunId, setSelectedRunId, selectedDataset, setSelectedDataset } = useAppState();
  const [runId, setRunId] = useState<string>(selectedRunId ?? "");
  const [dataset, setDataset] = useState<string>(selectedDataset ?? "");
  const [subject, setSubject] = useState<string>("");
  const [windowIndex, setWindowIndex] = useState<number>(initialWindowIndex);

  useEffect(() => {
    if (!dataset && experiments.length) {
      const run = experiments.find((e) => e.run_id === runId);
      setDataset(selectedDataset ?? run?.dataset ?? "");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [experiments]);

  const adapters = useApi(() => api.get<DatasetsListResponse>("/datasets"), []);
  useEffect(() => {
    if (!dataset && adapters.data?.adapters.length) setDataset(adapters.data.adapters[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adapters.data]);

  const subjects = useApi(
    () => api.get<DatasetSubjectsResponse>(`/datasets/${dataset}/subjects`),
    [dataset],
    { enabled: !!dataset },
  );
  useEffect(() => {
    setSubject("");
  }, [dataset]);

  const query = new URLSearchParams({
    dataset,
    window_index: String(windowIndex),
  });
  if (subject) query.set("subject", subject);
  if (runId) query.set("run_id", runId);

  const inspection = useApi(
    () => api.get<InspectResponse>(`/windows/inspect?${query.toString()}`, 180000),
    [dataset, subject, windowIndex, runId],
    { enabled: !!dataset },
  );

  const select = {
    dataset,
    setDataset: (v: string) => { setDataset(v); setSelectedDataset(v || null); },
    subject,
    setSubject,
    windowIndex,
    setWindowIndex,
    runId,
    setRunId: (v: string) => { setRunId(v); if (v) setSelectedRunId(v); },
  };

  return { select, inspection, subjects, adapters, experiments, selectedRunId, setSelectedRunId };
}
