// Screenshot of the viewer after scoring the mixed demo tile at stride 48, for the README.
//   node e2e/screenshot.mjs [out.png]       (needs `make up`)
import { chromium } from "@playwright/test";

const out = process.argv[2] ?? "../docs/img/viewer.png";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1 });
await page.goto(process.env.VIEWER_URL ?? "http://localhost:8088");
const tile = page.getByLabel("Demo tile");
const mixed = await tile.locator("option").filter({ hasText: "Mixed" }).getAttribute("value");
if (mixed !== null) await tile.selectOption(mixed);
await page.getByLabel("Window stride").selectOption("48");
await page.getByRole("button", { name: "Score tile" }).click();
await page.locator("#summary[data-state=scored]").waitFor({ timeout: 30_000 });
await page.screenshot({ path: out, fullPage: true });
await browser.close();
console.log(`wrote ${out}`);
