import { chromium } from "playwright-core";

const BASE = process.env.UI_URL ?? "http://localhost:5173";
const API = process.env.API_URL ?? "http://127.0.0.1:8420";
const OUT = "C:/Users/Admin/AppData/Local/Temp/opencode";

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 940 } });
page.on("pageerror", (e) => console.log(`[pageerror]`, e.message.slice(0, 200)));
await page.addInitScript(([apiUrl]) => {
  localStorage.setItem("thar.apiBase", apiUrl);
  localStorage.setItem("thar.theme", "dark");
}, [API]);

const shots = [
  ["overview", "#/overview", 14000],
  ["live", "#/live", 20000],
  ["windows", "#/windows", 60000],
  ["tda", "#/tda", 60000],
  ["evaluation", "#/evaluation", 20000],
  ["experiments", "#/experiments", 20000],
  ["model", "#/model", 20000],
  ["datasets", "#/datasets", 40000],
  ["system", "#/system", 15000],
];

for (const [name, hash, wait] of shots) {
  await page.goto(`${BASE}/${hash}`, { waitUntil: "networkidle", timeout: 90000 });
  await page.waitForTimeout(wait);
  await page.screenshot({ path: `${OUT}/shot-${name}.png`, fullPage: false });
  console.log(`shot-${name}.png done`);
}

// light theme + long scroll of tda
await page.goto(`${BASE}/#/tda`, { waitUntil: "networkidle" });
await page.evaluate(() => { localStorage.setItem("thar.theme", "light"); });
await page.waitForTimeout(45000);
await page.screenshot({ path: `${OUT}/shot-tda-light.png` });
console.log("shot-tda-light.png done");

await browser.close();
