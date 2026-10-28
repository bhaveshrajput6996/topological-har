import { chromium } from "playwright-core";

const UI = process.env.UI_URL ?? "http://localhost:5173";
const out = "C:/Users/Admin/AppData/Local/Temp/opencode";

const browser = await chromium.launch({ channel: "msedge", headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
await page.addInitScript(([apiUrl]) => {
  localStorage.setItem("thar.apiBase", apiUrl);
}, [process.env.API_URL ?? "http://127.0.0.1:8420"]);
const errors = [];
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });

await page.goto(UI + "/#/evaluation", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);

// 1. Confusion matrix labels visible (dark theme)
const cm = page.locator("svg").filter({ hasText: "Predicted" }).first();
await cm.scrollIntoViewIfNeeded().catch(() => {});
await page.screenshot({ path: `${out}/fix-eval-dark.png`, fullPage: false });

// 2. Click theme toggle -> light theme
await page.click('button[aria-label="Toggle theme"]');
await page.waitForTimeout(700);
const themeAttr = await page.evaluate(() => document.documentElement.dataset.theme);
const stored = await page.evaluate(() => localStorage.getItem("thar.theme"));
console.log("theme after toggle:", themeAttr, "| localStorage:", stored);
await page.screenshot({ path: `${out}/fix-eval-light.png`, fullPage: false });

// 3. Reload -> theme persists, toggle back works
await page.reload({ waitUntil: "networkidle" });
await page.waitForTimeout(1000);
const themeAfterReload = await page.evaluate(() => document.documentElement.dataset.theme);
console.log("theme after reload:", themeAfterReload);
await page.click('button[aria-label="Toggle theme"]');
await page.waitForTimeout(500);
const themeBack = await page.evaluate(() => document.documentElement.dataset.theme);
console.log("theme toggled back:", themeBack);

// 4. Light theme on TDA + overview
await page.goto(UI + "/#/tda", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await page.screenshot({ path: `${out}/fix-tda-light.png`, fullPage: false });
await page.goto(UI + "/#/", { waitUntil: "networkidle" });
await page.waitForTimeout(1500);
await page.screenshot({ path: `${out}/fix-overview-light.png`, fullPage: false });

console.log("console/page errors:", errors.length);
errors.slice(0, 5).forEach((e) => console.log("  " + e));
await browser.close();
