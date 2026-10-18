// @ts-check
import { expect, test } from "@playwright/test";

test("scores a demo tile through the Rust service and draws the heatmap", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Lymph node metastasis viewer" })).toBeVisible();
  await expect(page.getByLabel("Demo tile").locator("option")).not.toHaveCount(0);

  await page.getByRole("button", { name: "Score tile" }).click();
  const summary = page.locator("#summary");
  await expect(summary).toHaveAttribute("data-state", "scored", { timeout: 30_000 });
  await expect(summary).toContainText("8 x 8 at stride 96");
  await expect(summary).toContainText("Agreement with labels");

  // The overlay canvas has painted pixels after scoring.
  const painted = await page.locator("#overlay").evaluate((el) => {
    const canvas = /** @type {HTMLCanvasElement} */ (el);
    const data = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d")).getImageData(0, 0, canvas.width, canvas.height).data;
    let n = 0;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 0) n += 1;
    return n;
  });
  expect(painted).toBeGreaterThan(0);
});

test("overlapping stride gives a denser grid", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Window stride").selectOption("48");
  await page.getByRole("button", { name: "Score tile" }).click();
  await expect(page.locator("#summary")).toContainText("15 x 15 at stride 48", { timeout: 30_000 });
});
