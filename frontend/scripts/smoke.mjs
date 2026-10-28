/* Smoke test: drives the built dashboard against the live backend with Edge.
   All assertions are case-insensitive (CSS text-transform shows up in innerText). */
import { chromium } from "playwright-core";

const BASE = process.env.UI_URL ?? "http://localhost:5173";
const API = process.env.API_URL ?? "http://127.0.0.1:8420";

async function main() {
  const browser = await chromium.launch({ channel: "msedge", headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  const errors = [];
  const failedRequests = [];
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => {
    const url = r.url();
    if (!url.startsWith("data:")) failedRequests.push(`${url} :: ${r.failure()?.errorText}`);
  });

  await page.addInitScript(([apiUrl]) => {
    localStorage.setItem("thar.apiBase", apiUrl);
    localStorage.setItem("thar.theme", "dark");
  }, [API]);

  const results = [];
  async function check(name, fn, timeout = 45000) {
    try {
      await fn(timeout);
      results.push(`PASS ${name}`);
    } catch (e) {
      results.push(`FAIL ${name}: ${String(e).slice(0, 220)}`);
    }
  }

  const bodyText = () => page.evaluate(() => document.body.innerText.toLowerCase());
  const waitForText = async (needle, timeout) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeout) {
      const t = await bodyText();
      if (t.includes(needle.toLowerCase())) return;
      await page.waitForTimeout(400);
    }
    throw new Error(`text not found: ${needle}`);
  };

  /* ---------------- overview ---------------- */
  await page.goto(`${BASE}/#/overview`, { waitUntil: "networkidle", timeout: 60000 });
  await check("overview: backend chip online", () => waitForText("API online", 20000));
  await check("overview: pipeline stages", async () => {
    const t = await bodyText();
    for (const s of ["sensor data", "preprocessing", "windowing", "point cloud", "tda", "deepconvlstm", "fusion", "prediction"]) {
      if (!t.includes(s)) throw new Error(`missing stage ${s}`);
    }
  });
  await check("overview: run config card", () => waitForText("selected run configuration", 40000));
  await check("overview: transparency notes", async () => {
    const t = await bodyText();
    for (const s of ["subject-aware splitting", "pca", "fusion"]) if (!t.includes(s)) throw new Error(`missing note ${s}`);
  });
  await check("overview: quick predict", async () => {
    await page.click("text=Predict sample");
    await waitForText("truth:", 150000);
  }, 160000);

  /* ---------------- datasets ---------------- */
  await page.goto(`${BASE}/#/datasets`, { waitUntil: "networkidle" });
  await check("datasets: adapter tabs + info", async () => {
    await waitForText("uci_har", 60000);
    await waitForText("activity labels", 90000);
    await waitForText("walking", 30000);
  }, 120000);

  /* ---------------- window inspector ---------------- */
  await page.goto(`${BASE}/#/windows`, { waitUntil: "networkidle" });
  await check("windows: metadata + point cloud + pca note", async () => {
    await waitForText("window metadata", 180000);
    await waitForText("point cloud (pca projection)", 60000);
    const t = await bodyText();
    if (!t.includes("pca is visualization-only")) throw new Error("PCA note missing");
  }, 240000);

  /* ---------------- tda ---------------- */
  await page.goto(`${BASE}/#/tda`, { waitUntil: "networkidle" });
  await check("tda: diagram + image + betti", async () => {
    await waitForText("persistence diagram", 180000);
    await waitForText("betti & persistence statistics", 60000);
    const t = await bodyText();
    if (!t.includes("h0 — components")) throw new Error("H0 stats missing");
    if (!t.includes("h1 — loops")) throw new Error("H1 stats missing");
    if (!t.includes("persistence image")) throw new Error("image missing");
  }, 240000);

  /* ---------------- model ---------------- */
  await page.goto(`${BASE}/#/model`, { waitUntil: "networkidle" });
  await check("model: architecture diagrams", async () => {
    await waitForText("conv block 4", 40000);
    const t = await bodyText();
    if (!t.includes("concatenation")) throw new Error("fusion missing");
    if (!t.includes("lstm 2")) throw new Error("lstm missing");
  });

  /* ---------------- evaluation ---------------- */
  await page.goto(`${BASE}/#/evaluation`, { waitUntil: "networkidle" });
  await check("evaluation: metrics + confusion + split", async () => {
    await waitForText("test accuracy", 40000);
    await waitForText("confusion matrix", 40000);
    const t = await bodyText();
    if (!t.includes("subject-aware splitting")) throw new Error("split note missing");
    if (!t.includes("per-class metrics")) throw new Error("per-class missing");
  });

  /* ---------------- experiments ---------------- */
  await page.goto(`${BASE}/#/experiments`, { waitUntil: "networkidle" });
  await check("experiments: history table + detail", async () => {
    await waitForText("har-2026-003", 40000);
    await page.click("tr.clickable");
    await waitForText("split metadata (subject-aware)", 40000);
  });

  /* ---------------- live recognition ---------------- */
  await page.goto(`${BASE}/#/live`, { waitUntil: "networkidle" });
  await check("live: start replay → streaming predictions", async () => {
    await waitForText("start replay", 30000);
    await page.click("text=Start replay");
    await waitForText("windows streamed", 180000);
    await page.waitForFunction(
      () => [...document.querySelectorAll(".hero-value")].some((e) => /^[A-Z][A-Z _]+$/.test((e.textContent ?? "").trim())),
      { timeout: 180000 },
    );
    // prediction history rows appeared
    await page.waitForFunction(() => document.querySelectorAll("table tbody tr").length > 0, { timeout: 30000 });
    const t = await bodyText();
    if (!t.includes("ws connected")) throw new Error("ws not connected");
  }, 240000);

  /* ---------------- system ---------------- */
  await page.goto(`${BASE}/#/system`, { waitUntil: "networkidle" });
  await check("system: probes + request log", async () => {
    await page.click("text=Run probes");
    await waitForText("request log", 20000);
    await waitForText("experiment history", 30000);
  }, 80000);

  /* ---------------- responsive ---------------- */
  await page.setViewportSize({ width: 860, height: 900 });
  await page.goto(`${BASE}/#/overview`, { waitUntil: "networkidle" });
  await check("responsive: tablet overview no h-overflow", async () => {
    await page.waitForTimeout(900);
    const ok = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2);
    if (!ok) throw new Error("horizontal overflow at 860px");
  });
  await page.setViewportSize({ width: 1200, height: 800 });
  await page.goto(`${BASE}/#/overview`, { waitUntil: "networkidle" });
  await check("responsive: laptop overview no h-overflow", async () => {
    await page.waitForTimeout(600);
    const ok = await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2);
    if (!ok) throw new Error("horizontal overflow at 1200px");
  });

  console.log(results.join("\n"));
  const failures = results.filter((r) => r.startsWith("FAIL"));
  if (errors.length) console.log(`\nCONSOLE ERRORS (${errors.length}):`);
  for (const e of errors.slice(0, 12)) console.log("  " + e.slice(0, 240));
  if (failedRequests.length) console.log(`\nFAILED REQUESTS (${failedRequests.length}):`);
  for (const r of failedRequests.slice(0, 10)) console.log("  " + r.slice(0, 220));
  await browser.close();
  process.exit(failures.length || errors.length ? 1 : 0);
}

main().catch((e) => {
  console.error("SMOKE CRASH:", e);
  process.exit(2);
});
