import { useEffect, useState } from "react";
import { Badge, Button, Card, ErrorState, LoadingBlock, SectionNote, StatCard } from "../components/ui";
import { DatasetSubjectPicker, RunPicker } from "../components/pickers";
import { useInspection } from "../components/useInspection";
import { ScatterPlot, timeColor } from "../components/charts/ScatterPlot";
import { Heatmap } from "../components/charts/Heatmap";
import { LineChart } from "../components/charts/LineChart";
import { fmtMs, fmtNum, prettyLabel } from "../lib/format";

export default function TdaAnalysis() {
  const { select, inspection, subjects, adapters, experiments } = useInspection();
  const insp = inspection.data;
  const pd = insp?.persistence_diagram;
  const pi = insp?.persistence_image;
  const betti = insp?.betti_statistics;
  const [showVector, setShowVector] = useState(false);
  const [showBars, setShowBars] = useState(true);

  // persistence bars per dimension: sorted persistence values (log scale-ish bar chart)
  const persistSeries = (() => {
    if (!pd) return null;
    const out: Record<string, { x: number; y: number }[]> = {};
    for (const key of Object.keys(pd.api)) {
      const rows = pd.api[key].map(([b, d]) => d - b).sort((a, b) => b - a);
      out[key] = rows.map((p, i) => ({ x: i, y: p }));
    }
    return out;
  })();

  const h0Points = (pd?.api["H0"] ?? []).map(([b, d]) => ({ x: b, y: d, color: "var(--blue)", label: `H0 · birth ${b.toFixed(3)}`, sub: `persistence ${(d - b).toFixed(3)}` }));
  const h1Points = (pd?.api["H1"] ?? []).map(([b, d]) => ({ x: b, y: d, color: "var(--violet)", label: `H1 · birth ${b.toFixed(3)}`, sub: `persistence ${(d - b).toFixed(3)}` }));
  const cloudPoints = (insp?.point_cloud.pcaviz2 ?? []).map(([x, y], i) => ({
    x, y, color: timeColor(i, Math.max(1, (insp?.point_cloud.pcaviz2.length ?? 2) - 1)),
    label: `sample ${insp?.point_cloud.point_indices[i] ?? i}`,
  }));

  const stat = (k: string) => (betti ? betti[k] ?? null : null);

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">TDA Analysis</div>
          <div className="page-desc">
            Topological branch of the pipeline: the standardized sensor window is a point cloud
            (one point per time step, full channel dimensionality). A Vietoris–Rips complex is
            built over it and persistent homology in H0 / H1 is computed with GUDHI.
          </div>
        </div>
      </div>

      <Card title="Window selection">
        <DatasetSubjectPicker
          datasets={adapters.data?.adapters ?? []}
          dataset={select.dataset}
          onDataset={select.setDataset}
          subjects={subjects.data?.subjects ?? []}
          subject={select.subject}
          onSubject={select.setSubject}
          windowIndex={select.windowIndex}
          onWindowIndex={select.setWindowIndex}
        />
        <RunPicker
          experiments={experiments}
          value={select.runId || null}
          onChange={select.setRunId}
          label="Attach run (adds prediction)"
          style={{ minWidth: 240 }}
        />
      </Card>

      {inspection.loading ? (
        <div style={{ marginTop: 14 }}><Card><LoadingBlock label="Computing Vietoris–Rips persistent homology…" /></Card></div>
      ) : inspection.error ? (
        <div style={{ marginTop: 14 }}><Card title="TDA Analysis"><ErrorState error={inspection.error} context="TDA computation" retry={inspection.reload} /></Card></div>
      ) : insp && pd && pi && betti ? (
        <>
          {/* headline stats */}
          <div className="grid grid-c4" style={{ marginTop: 14 }}>
            <StatCard label="H0 components" accent="var(--blue)" value={fmtNum(stat("h0_count"), 0)} sub="connected components" />
            <StatCard label="H1 loops" accent="var(--violet)" value={fmtNum(stat("h1_count"), 0)} sub="1-dimensional loops" />
            <StatCard label="Scale reference" accent="var(--cyan)" value={fmtNum(insp.persistence_diagram.scale_reference, 3)} sub="cloud diameter used to clip essential features" />
            <StatCard label="Feature vector" accent="var(--amber)" value={`${pi.vector.length} dims`} sub={`image ${pi.resolution[0]}×${pi.resolution[1]} · σ=${pi.bandwidth}`} />
          </div>

          <div className="grid grid-c2" style={{ marginTop: 14 }}>
            {/* diagram */}
            <Card
              title="Persistence diagram"
              sub="Each point is a topological feature: birth on x, death on y. Distance above the diagonal = persistence."
            >
              <div className="row" style={{ marginBottom: 8 }}>
                <Badge tone="cyan">H0 — {h0Points.length} pts</Badge>
                <Badge tone="violet">H1 — {h1Points.length} pts</Badge>
                {insp.persistence_diagram.n_essential["0"] > 0 && <Badge tone="amber">{insp.persistence_diagram.n_essential["0"]} essential H0</Badge>}
              </div>
              <ScatterPlot
                points={[...h0Points, ...h1Points]}
                height={280}
                xLabel="birth"
                yLabel="death"
                diagonal
              />
            </Card>

            {/* persistence image */}
            <Card
              title="Persistence image"
              sub="Vectorized topology embedding fed to the fusion head (heatmap of the [0,1]² grid)"
            >
              <Heatmap values={pi.vector} rows={pi.resolution[0]} cols={pi.resolution[1]} height={280} />
              <div className="row" style={{ marginTop: 8, justifyContent: "space-between" }}>
                <div className="hint">
                  resolution {pi.resolution[0]}×{pi.resolution[1]} · bandwidth {pi.bandwidth} · vector {pi.vector.length} dims
                </div>
                <div className="row" style={{ gap: 6 }}>
                  <Button size="sm" variant="ghost" onClick={() => setShowBars((s) => !s)}>
                    {showBars ? "Hide" : "Show"} persistence bars
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => setShowVector((s) => !s)}>
                    {showVector ? "Hide" : "Expand"} raw vector
                  </Button>
                </div>
              </div>
            </Card>
          </div>

          {persistSeries && (
            <div className="grid grid-c2" style={{ marginTop: 0 }}>
              {showBars && Object.keys(persistSeries).map((key) => (
                <Card key={key} title={`${key} persistence values`} sub="Sorted persistence (death − birth) of each feature">
                  <LineChart
                    series={[{ name: key, color: key === "H0" ? "var(--blue)" : "var(--violet)", points: persistSeries[key] }]}
                    height={180}
                    xLabel="feature rank"
                    yLabel="persistence"
                    area
                  />
                </Card>
              ))}
            </div>
          )}

          {/* betti / persistence statistics */}
          <Card title="Betti & persistence statistics" sub="Computed by the backend from the persistence diagram" style={{ marginTop: 14 }}>
            <div className="grid grid-c2">
              <div className="card tight">
                <div className="card-title" style={{ color: "var(--blue)", marginBottom: 10 }}>H0 — components</div>
                <div className="grid grid-c2" style={{ gap: 8 }}>
                  <MiniStat label="Count" v={stat("h0_count")} fmt={(v) => fmtNum(v, 0)} />
                  <MiniStat label="Total persistence" v={stat("h0_total_persistence")} fmt={(v) => fmtNum(v, 3)} />
                  <MiniStat label="Max persistence" v={stat("h0_max_persistence")} fmt={(v) => fmtNum(v, 4)} />
                  <MiniStat label="Mean persistence" v={stat("h0_mean_persistence")} fmt={(v) => fmtNum(v, 4)} />
                  <MiniStat label="Entropy" v={stat("h0_entropy")} fmt={(v) => fmtNum(v, 3)} accent="var(--cyan)" />
                </div>
              </div>
              <div className="card tight">
                <div className="card-title" style={{ color: "var(--violet)", marginBottom: 10 }}>H1 — loops</div>
                <div className="grid grid-c2" style={{ gap: 8 }}>
                  <MiniStat label="Count" v={stat("h1_count")} fmt={(v) => fmtNum(v, 0)} />
                  <MiniStat label="Total persistence" v={stat("h1_total_persistence")} fmt={(v) => fmtNum(v, 3)} />
                  <MiniStat label="Max persistence" v={stat("h1_max_persistence")} fmt={(v) => fmtNum(v, 4)} />
                  <MiniStat label="Mean persistence" v={stat("h1_mean_persistence")} fmt={(v) => fmtNum(v, 4)} />
                  <MiniStat label="Entropy" v={stat("h1_entropy")} fmt={(v) => fmtNum(v, 3)} accent="var(--violet)" />
                </div>
              </div>
            </div>
          </Card>

          {/* raw vector */}
          {showVector && (
            <Card title="Raw persistence image vector" sub={`${pi.vector.length} float64 values — row-major over the ${pi.resolution[0]}×${pi.resolution[1]} grid`} style={{ marginTop: 14 }}>
              <div style={{ maxHeight: 260, overflowY: "auto", fontFamily: "var(--mono)", fontSize: 11, lineHeight: 1.7, color: "var(--text-dim)", wordBreak: "break-all" }}>
                {pi.vector.map((v, i) => (
                  <span key={i} style={{ marginRight: 10, color: v > 0 ? "var(--text)" : "var(--muted)" }}>
                    {v.toFixed(4)}
                  </span>
                ))}
              </div>
            </Card>
          )}

          {/* explanations */}
          <div className="grid grid-c2" style={{ marginTop: 14 }}>
            <Card title="What you are looking at">
              <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                <SectionNote tone="violet">
                  <span><b>Point cloud.</b> Each of the subsampled time steps becomes a point with
                  all channel values as coordinates ({insp.point_cloud.n_dims}-D). The backend keeps the
                  full resolution for the deep branch and only subsamples for the TDA branch.</span>
                </SectionNote>
                <SectionNote tone="violet">
                  <span><b>H0 / H1.</b> H0 features track connected components merging as the
                  distance scale grows; H1 features track loops. Persistence = death − birth:
                  long-lived features indicate robust structure.</span>
                </SectionNote>
              </div>
            </Card>
            <Card title="Pipeline honesty">
              <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
                <SectionNote>
                  <span><b>PCA is visualization-only.</b> The scatter on this page shows a 2-D
                  projection; TDA ran on the full-dimensional point cloud.</span>
                </SectionNote>
                <SectionNote>
                  <span><b>Subsampling.</b> Shown cloud has {insp.point_cloud.n_points} of the window's{" "}
                  {insp.window_shape[0]} samples (stride method) — an ablation knob for TDA latency,
                  not a claim about the full window.</span>
                </SectionNote>
              </div>
            </Card>
          </div>
        </>
      ) : null}
    </>
  );
}

function MiniStat({
  label, v, fmt, accent,
}: { label: string; v: number | null; fmt: (x: number | null) => string; accent?: string }) {
  return (
    <div style={{ background: "var(--panel-2)", border: "1px solid var(--border)", borderRadius: 8, padding: "8px 12px" }}>
      <div className="hint" style={{ textTransform: "uppercase", fontSize: 9.5, letterSpacing: "0.1em" }}>{label}</div>
      <div className="mono" style={{ fontSize: 14.5, marginTop: 3, color: accent ?? "var(--text)" }}>{fmt(v)}</div>
    </div>
  );
}
