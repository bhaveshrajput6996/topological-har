# TOPOLOGICAL HAR — Research Dashboard

React (Vite + TypeScript) frontend for the TOPOLOGICAL HAR research project.
It is a pure client for the FastAPI backend in `../backend` — the backend is
never modified by this app, and every number shown comes from a live API
response. Missing data is shown as "N/A" / "Data unavailable", never fabricated.

## Prerequisites

- Node.js 18+ (built with Node 24)
- Python backend (`../backend`, see its own docs)

## Run

**1. Start the backend** (from the project root):

```powershell
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8000
```

**2. Start the frontend dev server:**

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (default `http://localhost:5173` — use `localhost`,
not `127.0.0.1`, since Vite binds to IPv6 `::1`).

## API URL configuration

The app talks to the backend at, in order of precedence:

1. localStorage key `thar.apiBase` (editable on the **System / API Status**
   page — takes effect immediately)
2. `VITE_API_URL` environment variable (see `.env.development`)
3. Default `http://127.0.0.1:8000`

If you start the backend on another port, either set `VITE_API_URL` or enter
the base URL on the System page.

## Pages

| Page | Purpose |
| --- | --- |
| Overview | Active run summary, pipeline diagram, quick single-window predict |
| Live Recognition | Replay-driven streaming recognition (WebSocket), pause/step/stop |
| Dataset Explorer | Adapters, download/load progress, subject listing |
| Window Inspector | Per-window sensor signal, point cloud, PCA 2D projection |
| TDA Analysis | Persistence diagram, persistence image, Betti statistics |
| Model Analysis | Architecture (DeepConvLSTM, TDA branch, fusion) + run config |
| Evaluation | Aggregate metrics, per-class metrics, confusion matrix, training curves, split/leakage record |
| Experiments | Experiment history + run detail; inline launcher posting to `POST /experiments` |
| System / API Status | Backend health probe, endpoint probes, API base override, request log |

## Notes on data honesty

- `GET /experiments` rows use `-1` as a sentinel for "metric not stored";
  the UI maps `-1` to `null` and renders "N/A".
- Replay has no server-side pause; the pause button freezes the client-side
  display only (server keeps streaming), which the UI states explicitly.
- Dataset adapters that are not downloaded yet show their real state from
  `GET /datasets`; nothing is simulated.

## Verification

```powershell
npx tsc --noEmit      # type check
npm run build         # production build (dist/)
node scripts\smoke.mjs  # 15-check headless-browser smoke test
```

The smoke test needs both servers running and honors `UI_URL` / `API_URL`
environment variables (defaults: `http://localhost:5173` and
`http://127.0.0.1:8000`). `scripts/shots.mjs` captures page screenshots,
`scripts/theme-check.mjs` verifies the theme toggle.
