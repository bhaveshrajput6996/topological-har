import { useCallback, useState } from "react";
import { api, getApiBase, getRequestLog, getLastSuccessTs, setApiBase, envBase, type RequestLogEntry } from "../api/client";
import { useInterval } from "../api/hooks";
import { useAppState } from "../state/AppState";
import { Badge, Button, Card, Field, SectionNote, StatCard, StatusDot } from "../components/ui";
import { relTime } from "../lib/format";

interface ProbeResult {
  name: string;
  endpoint: string;
  status: number;
  ms: number;
  ok: boolean;
  info: string;
}

const PROBES: { name: string; endpoint: string }[] = [
  { name: "Health", endpoint: "/health" },
  { name: "Dataset registry", endpoint: "/datasets" },
  { name: "Experiment history", endpoint: "/experiments" },
];

export default function SystemStatus() {
  const { backendStatus, health, lastHealthCheck, checkBackend } = useAppState();
  const [base, setBase] = useState(getApiBase());
  const [probes, setProbes] = useState<ProbeResult[] | null>(null);
  const [probing, setProbing] = useState(false);
  const [wsProbe, setWsProbe] = useState<{ status: string; info: string } | null>(null);
  const [, setLogTick] = useState(0);

  // refresh the request-log table periodically (client-side ring buffer)
  useInterval(() => setLogTick((t) => t + 1), 2000);

  const runProbes = useCallback(async () => {
    setProbing(true);
    const results: ProbeResult[] = [];
    for (const p of PROBES) {
      const t0 = performance.now();
      try {
        const res = await fetch(`${getApiBase()}${p.endpoint}`);
        const ms = performance.now() - t0;
        const body = res.ok ? await res.json().catch(() => null) : null;
        results.push({
          name: p.name, endpoint: p.endpoint, status: res.status, ms: Math.round(ms), ok: res.ok,
          info: p.endpoint === "/health" && body ? `v${body.version} · CUDA ${body.torch_cuda}` : "ok",
        });
      } catch {
        results.push({
          name: p.name, endpoint: p.endpoint, status: 0, ms: Math.round(performance.now() - t0), ok: false,
          info: "unreachable",
        });
      }
    }
    setProbes(results);
    setProbing(false);
  }, []);

  const probeWs = useCallback(() => {
    setWsProbe({ status: "connecting", info: "" });
    try {
      const base = getApiBase().replace(/^http/, "ws");
      const ws = new WebSocket(`${base}/ws/replay/probe-invalid-session`);
      const done = (status: string, info: string) => {
        setWsProbe({ status, info });
        try { ws.close(); } catch { /* ignore */ }
      };
      ws.onopen = () => done("connected", "handshake accepted (upgrade ok)");
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(String(e.data)) as { error?: string };
          done("responding", msg.error ? `endpoint alive — server replied: ${msg.error}` : "endpoint alive");
        } catch {
          done("responding", "endpoint alive");
        }
      };
      ws.onerror = () => done("error", "WebSocket handshake failed");
      setTimeout(() => {
        if (ws.readyState === WebSocket.CONNECTING) done("timeout", "no response within 5 s");
      }, 5000);
    } catch (e) {
      setWsProbe({ status: "error", info: (e as Error).message });
    }
  }, []);

  const log: readonly RequestLogEntry[] = getRequestLog();
  const lastOk = getLastSuccessTs();

  return (
    <>
      <div className="page-head">
        <div>
          <div className="page-title">System / API Status</div>
          <div className="page-desc">
            Live connectivity diagnostics. The backend URL can be overridden below — it is stored in
            this browser only.
          </div>
        </div>
        <Button variant="primary" onClick={() => void checkBackend()}>Re-check backend</Button>
      </div>

      <div className="grid grid-c4">
        <StatCard
          label="Backend"
          accent={backendStatus === "online" ? "var(--green)" : backendStatus === "offline" ? "var(--red)" : "var(--amber)"}
          value={backendStatus === "online" ? "Reachable" : backendStatus === "offline" ? "Unreachable" : "Checking…"}
          sub={lastHealthCheck ? relTime(lastHealthCheck) : ""}
        />
        <StatCard label="API version" accent="var(--cyan)" value={health?.version ?? "N/A"} sub={health ? `CUDA: ${health.torch_cuda}` : undefined} />
        <StatCard label="Last successful request" accent="var(--green)" value={lastOk ? relTime(lastOk) : "—"} sub="any REST endpoint" />
        <StatCard label="API base URL" accent="var(--violet)" value={<span style={{ fontSize: 13 }}>{base}</span>} sub={base === envBase() ? "from VITE_API_URL / default" : "browser override"} />
      </div>

      <div className="grid grid-c2" style={{ marginTop: 14 }}>
        <Card title="Endpoint probes" sub="Measured round-trip time of representative GET endpoints">
          <div className="row" style={{ marginBottom: 10 }}>
            <Button size="sm" variant="primary" disabled={probing} onClick={() => void runProbes()}>
              {probing ? "Probing…" : probes ? "Re-run probes" : "Run probes"}
            </Button>
            <Button size="sm" variant="ghost" onClick={probeWs}>Probe WebSocket</Button>
          </div>
          {probes ? (
            <table className="table">
              <thead><tr><th>Probe</th><th>Endpoint</th><th className="num">HTTP</th><th className="num">RTT</th><th>Info</th></tr></thead>
              <tbody>
                {probes.map((p) => (
                  <tr key={p.endpoint}>
                    <td style={{ display: "flex", alignItems: "center", gap: 6 }}><StatusDot status={p.ok ? "ok" : "err"} />{p.name}</td>
                    <td>{p.endpoint}</td>
                    <td className="num">{p.status || "—"}</td>
                    <td className="num">{p.ms} ms</td>
                    <td className="wrap">{p.info}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="hint" style={{ padding: 8 }}>Run the probes to measure endpoint health and latency.</div>
          )}
          {wsProbe && (
            <div style={{ marginTop: 10 }}>
              <SectionNote tone={wsProbe.status === "responding" || wsProbe.status === "connected" ? "green" : "amber"}>
                <span><b>WebSocket /ws/replay:</b> {wsProbe.status} — {wsProbe.info}</span>
              </SectionNote>
            </div>
          )}
        </Card>

        <Card title="Connection settings" sub="Environment-based backend URL with browser override">
          <Field label="API base URL">
            <input
              className="input mono"
              value={base}
              onChange={(e) => setBase(e.target.value)}
              placeholder={envBase()}
            />
          </Field>
          <div className="row" style={{ marginTop: 10 }}>
            <Button size="sm" variant="primary" onClick={() => { setApiBase(base); void checkBackend(); }}>
              Apply & reconnect
            </Button>
            <Button size="sm" variant="ghost" onClick={() => { setApiBase(""); setBase(getApiBase()); void checkBackend(); }}>
              Reset to default
            </Button>
          </div>
          <div style={{ marginTop: 12 }}>
            <SectionNote>
              <span>Default comes from <span className="mono">VITE_API_URL</span> (see
              <span className="mono"> .env.development</span>), falling back to
              <span className="mono"> http://127.0.0.1:8000</span>. The backend enables permissive
              CORS, so any local port works.</span>
            </SectionNote>
          </div>
          <div className="divider" />
          <div className="hint">
            Expected endpoints — REST: <span className="mono">/health /datasets /experiments /predict /windows/inspect /replay/*</span> ·
            WS: <span className="mono">/ws/replay/&#123;session_id&#125;</span>
          </div>
        </Card>
      </div>

      <Card
        title="Request log"
        sub="Client-side ring buffer of the dashboard's own API traffic (most recent first)"
        style={{ marginTop: 14 }}
      >
        {log.length === 0 ? (
          <div className="hint" style={{ padding: 10 }}>No requests recorded yet in this session.</div>
        ) : (
          <div style={{ maxHeight: 320, overflowY: "auto" }}>
            <table className="table">
              <thead><tr><th>Time</th><th>Method</th><th>Endpoint</th><th className="num">HTTP</th><th className="num">RTT</th><th /></tr></thead>
              <tbody>
                {log.map((r, i) => (
                  <tr key={i}>
                    <td>{new Date(r.ts).toLocaleTimeString()}</td>
                    <td>{r.method}</td>
                    <td style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis" }}>{r.endpoint}</td>
                    <td className="num">{r.status || "—"}</td>
                    <td className="num">{r.ms} ms</td>
                    <td>{r.ok ? <Badge tone="green">ok</Badge> : <Badge tone="red">failed</Badge>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </>
  );
}
