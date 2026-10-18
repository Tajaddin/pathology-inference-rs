// @ts-check
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { PathologyApiError, PathologyClient, heatColor } from "../src/index.js";

/**
 * A fake fetch that answers from a queue and records requests.
 * @param {Array<{ status: number, body?: unknown, headers?: Record<string, string> } | Error>} answers
 */
function fakeFetch(answers) {
  /** @type {Array<{ url: string, init: RequestInit }>} */
  const calls = [];
  /** @type {typeof fetch} */
  const fn = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    const next = answers.shift();
    if (!next) throw new Error("no more answers");
    if (next instanceof Error) throw next;
    return new Response(next.body === undefined ? "" : JSON.stringify(next.body), {
      status: next.status,
      headers: { "content-type": "application/json", ...(next.headers ?? {}) },
    });
  };
  return { fn, calls };
}

const ok = { model_version: "1", threshold: 0.5, results: [{ id: "a", probability: 0.9, tumor: true }] };
const noSleep = async () => {};

describe("PathologyClient", () => {
  it("sends the key, the JSON body and returns the parsed response", async () => {
    const { fn, calls } = fakeFetch([{ status: 200, body: ok }]);
    const client = new PathologyClient({ baseUrl: "http://svc/", apiKey: "k".repeat(20), fetch: fn });
    const res = await client.classify([{ id: "a", imageBase64: "AAAA" }]);
    assert.deepEqual(res, ok);
    assert.equal(calls[0].url, "http://svc/v1/classify");
    assert.equal(/** @type {Record<string, string>} */ (calls[0].init.headers).authorization, `Bearer ${"k".repeat(20)}`);
    assert.deepEqual(JSON.parse(String(calls[0].init.body)), { patches: [{ id: "a", image_base64: "AAAA" }] });
  });

  it("retries 503 and 429, honouring Retry-After", async () => {
    const { fn, calls } = fakeFetch([
      { status: 503, body: { error: { code: "unavailable", message: "loading" } } },
      { status: 429, body: { error: { code: "rate_limited" } }, headers: { "retry-after": "2" } },
      { status: 200, body: ok },
    ]);
    /** @type {number[]} */
    const waits = [];
    const client = new PathologyClient({
      baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: fn, retries: 2,
      sleep: async (ms) => { waits.push(ms); },
    });
    assert.deepEqual(await client.classify([{ imageBase64: "AAAA" }]), ok);
    assert.equal(calls.length, 3);
    assert.deepEqual(waits, [200, 2000]);
  });

  it("does not retry client errors and exposes the service's code", async () => {
    const { fn, calls } = fakeFetch([{ status: 422, body: { error: { code: "invalid_request", message: "patches must be 96x96" } } }]);
    const client = new PathologyClient({ baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: fn, sleep: noSleep });
    await assert.rejects(client.classify([{ imageBase64: "AAAA" }]), (err) => {
      assert.ok(err instanceof PathologyApiError);
      assert.equal(err.status, 422);
      assert.equal(err.code, "invalid_request");
      assert.equal(err.message, "patches must be 96x96");
      assert.equal(err.retryable, false);
      return true;
    });
    assert.equal(calls.length, 1);
  });

  it("gives up after the configured retries on network errors", async () => {
    const { fn, calls } = fakeFetch([new TypeError("fetch failed"), new TypeError("fetch failed")]);
    const client = new PathologyClient({ baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: fn, retries: 1, sleep: noSleep });
    await assert.rejects(client.model(), (err) => err instanceof PathologyApiError && err.code === "network");
    assert.equal(calls.length, 2);
  });

  it("times out slow requests", async () => {
    /** @type {typeof fetch} */
    const slow = (_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    });
    const client = new PathologyClient({ baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: slow, timeoutMs: 20, retries: 0 });
    await assert.rejects(client.heatmap("AAAA"), /timed out after 20 ms/);
  });

  it("validates arguments before calling the service", async () => {
    const { fn, calls } = fakeFetch([]);
    const client = new PathologyClient({ baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: fn });
    await assert.rejects(client.classify([]), RangeError);
    await assert.rejects(client.classify(Array(65).fill({ imageBase64: "A" })), RangeError);
    assert.throws(() => new PathologyClient({ baseUrl: "", apiKey: "x" }), TypeError);
    assert.equal(calls.length, 0);
  });

  it("sends the stride only when given", async () => {
    const heat = { model_version: "1", threshold: 0.5, patch_size: 96, stride: 48, rows: 1, cols: 1, probabilities: [0.2], max_probability: 0.2, windows_above_threshold: 0 };
    const { fn, calls } = fakeFetch([{ status: 200, body: heat }, { status: 200, body: heat }]);
    const client = new PathologyClient({ baseUrl: "http://svc", apiKey: "key-0000000000000001", fetch: fn });
    await client.heatmap("AAAA", { stride: 48 });
    await client.heatmap("AAAA");
    assert.deepEqual(JSON.parse(String(calls[0].init.body)), { image_base64: "AAAA", stride: 48 });
    assert.deepEqual(JSON.parse(String(calls[1].init.body)), { image_base64: "AAAA" });
  });
});

describe("heatColor", () => {
  it("is transparent below the floor and red at 1", () => {
    assert.deepEqual(heatColor(0.05), [0, 0, 0, 0]);
    assert.deepEqual(heatColor(Number.NaN), [0, 0, 0, 0]);
    assert.deepEqual(heatColor(1), [255, 0, 0, 220]);
    const mid = heatColor(0.55);
    assert.ok(mid[1] > 0 && mid[1] < 220 && mid[3] > 80 && mid[3] < 220);
  });
});
