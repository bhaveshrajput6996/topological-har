import { useEffect, useState } from "react";
import { api } from "../api/client";
import { useApi } from "../api/hooks";
import type { DatasetInfo } from "../api/types";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, ErrorState, LoadingBlock, SectionNote, Skeleton, StatusDot } from "../components/ui";
import { prettyLabel } from "../lib/format";

export default function DatasetExplorer() {
  const { selectedDataset, setSelectedDataset } = useAppState();
  const adapters = useApi(() => api.get<{ adapters: string[] }>("/datasets"), []);
  const [active, setActive] = useState<string | null>(selectedDataset ?? null);

  useEffect(() => {
    if (!active && adapters.data?.adapters.length) {
      setActive(selectedDataset ?? adapters.data.adapters[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adapters.data]);

  const info = useApi(
    () => api.post<DatasetInfo>("/datasets/load", { name: active }),
    [active],
    { enabled: !!active },
  );

  if (adapters.loading) return <Skeleton h={300} />;
  if (adapters.error) {
    return (
      <Card title="Dataset Explorer">
        <ErrorState error={adapters.error} context="Dataset registry" retry={adapters.reload} />
      </Card>
    );
  }

  const d = info.data;

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">Dataset Explorer</div>
          <div className="page-desc">
            Dataset adapters served by the backend. Statistics below are loaded live from
            <span className="mono"> POST /datasets/load</span> — nothing is estimated in the UI.
          </div>
        </div>
      </div>

      <div className="row" style={{ marginBottom: 14 }}>
        {(adapters.data?.adapters ?? []).map((name) => (
          <button
            key={name}
            className={`btn ${active === name ? "primary" : ""}`}
            onClick={() => setActive(name)}
          >
            {name}
          </button>
        ))}
      </div>

      <div className="grid grid-c4" style={{ marginBottom: 14 }}>
        <div className="stat">
          <div className="stat-label"><span className="dot" style={{ background: "var(--blue)" }} />Dataset</div>
          {info.loading ? <div className="skeleton" style={{ height: 22, marginTop: 9, width: "60%" }} /> : (
            <div className="stat-value sm" style={{ color: "var(--blue)" }}>{d?.name ?? "—"}</div>
          )}
          {d && <div className="stat-sub mono">{d.kind}</div>}
        </div>
        <div className="stat">
          <div className="stat-label"><span className="dot" style={{ background: "var(--cyan)" }} />Subjects</div>
          {info.loading ? <div className="skeleton" style={{ height: 22, marginTop: 9, width: "40%" }} /> : (
            <div className="stat-value" style={{ color: "var(--cyan)" }}>{d?.subjects.length ?? "—"}</div>
          )}
          {d?.subjects && d.subjects.length > 0 && (
            <div className="stat-sub">{d.subjects.slice(0, 3).join(", ")}{d.subjects.length > 3 ? " …" : ""}</div>
          )}
        </div>
        <div className="stat">
          <div className="stat-label"><span className="dot" style={{ background: "var(--green)" }} />Activities</div>
          {info.loading ? <div className="skeleton" style={{ height: 22, marginTop: 9, width: "40%" }} /> : (
            <div className="stat-value" style={{ color: "var(--green)" }}>{d?.label_names.length ?? "—"}</div>
          )}
          {d && <div className="stat-sub">{d.label_names.slice(0, 3).map((l) => prettyLabel(l)).join(", ")}{d.label_names.length > 3 ? " …" : ""}</div>}
        </div>
        <div className="stat">
          <div className="stat-label"><span className="dot" style={{ background: "var(--violet)" }} />Sampling rate</div>
          {info.loading ? <div className="skeleton" style={{ height: 22, marginTop: 9, width: "50%" }} /> : (
            <div className="stat-value" style={{ color: "var(--violet)" }}>{d ? `${d.sampling_rate} Hz` : "—"}</div>
          )}
          {d && <div className="stat-sub">{d.channel_names.length} channels</div>}
        </div>
      </div>

      <div className="grid grid-c2">
        <Card title="Dataset details">
          {info.loading ? (
            <LoadingBlock label="Loading dataset (first load reads all signal files)…" />
          ) : info.error ? (
            <ErrorState error={info.error} context={`Dataset ${active}`} retry={info.reload} />
          ) : d ? (
            <>
              <dl className="kv">
                <dt>name</dt><dd>{d.name}</dd>
                <dt>kind</dt><dd>{d.kind}</dd>
                <dt>sampling rate</dt><dd>{d.sampling_rate} Hz</dd>
                <dt>source path</dt><dd>{d.source_path}</dd>
              </dl>
              <div className="divider" />
              <div className="hint" style={{ marginBottom: 6 }}>Channels</div>
              <div className="chip-list" style={{ marginBottom: 12 }}>
                {d.channel_names.map((c) => <span key={c} className="chip">{c}</span>)}
              </div>
              <div className="hint" style={{ marginBottom: 6 }}>Activity labels</div>
              <div className="chip-list">
                {d.label_names.map((l) => <span key={l} className="chip acc">{prettyLabel(l)}</span>)}
              </div>
              <div className="divider" />
              <div style={{ display: "flex", gap: 8 }}>
                <Button
                  size="sm"
                  variant={selectedDataset === d.name ? "primary" : ""}
                  onClick={() => { setSelectedDataset(d.name); setActive(d.name); }}
                >
                  {selectedDataset === d.name ? "✓ Active dataset" : "Set as active dataset"}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => info.reload()}>Reload info</Button>
              </div>
            </>
          ) : null}
        </Card>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Card title="Subjects" sub="Subject IDs as normalized by the adapter">
            {info.loading ? (
              <Skeleton h={120} />
            ) : info.error ? null : d ? (
              <div className="chip-list">
                {d.subjects.map((s) => <span key={s} className="chip">{s}</span>)}
              </div>
            ) : null}
          </Card>
          <Card title="Adapter notes" sub="Verbatim from the backend dataset object">
            {info.loading || !d ? (
              <Skeleton h={80} />
            ) : (
              <SectionNote>
                <span>{d.notes}</span>
              </SectionNote>
            )}
            {d && (
              <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
                <Badge tone={d.kind === "presegmented" ? "violet" : "cyan"}>
                  <StatusDot status="ok" />
                  {d.kind === "presegmented" ? "windows provided by dataset" : d.kind}
                </Badge>
                {d.name === "uci_har" && <Badge>2.56 s · 50% overlap (dataset-defined)</Badge>}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
