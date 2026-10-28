import { lazy, Suspense } from "react";
import { useRoute } from "./state/router";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { LoadingBlock } from "./components/ui";

const Overview = lazy(() => import("./pages/Overview"));
const LiveRecognition = lazy(() => import("./pages/LiveRecognition"));
const DatasetExplorer = lazy(() => import("./pages/DatasetExplorer"));
const WindowInspector = lazy(() => import("./pages/WindowInspector"));
const TdaAnalysis = lazy(() => import("./pages/TdaAnalysis"));
const ModelAnalysis = lazy(() => import("./pages/ModelAnalysis"));
const Evaluation = lazy(() => import("./pages/Evaluation"));
const Experiments = lazy(() => import("./pages/Experiments"));
const SystemStatus = lazy(() => import("./pages/SystemStatus"));

export default function App() {
  const route = useRoute();
  const key = `${route.page}|${route.params.toString()}`;
  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-col">
        <Header />
        <main className="content">
          <div className="page">
            <Suspense key={key} fallback={<LoadingBlock label="Loading page…" />}>
              <Page route={route.page} />
            </Suspense>
          </div>
        </main>
      </div>
    </div>
  );
}

function Page({ route: page }: { route: string }) {
  switch (page) {
    case "overview": return <Overview />;
    case "live": return <LiveRecognition />;
    case "datasets": return <DatasetExplorer />;
    case "windows": return <WindowInspector />;
    case "tda": return <TdaAnalysis />;
    case "model": return <ModelAnalysis />;
    case "evaluation": return <Evaluation />;
    case "experiments": return <Experiments />;
    case "system": return <SystemStatus />;
    default:
      return (
        <div className="state-block" style={{ padding: 80 }}>
          <div className="big">Unknown page “{page}”</div>
          <div className="hint">Use the sidebar to navigate.</div>
        </div>
      );
  }
}
