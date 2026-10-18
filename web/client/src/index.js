// @ts-check
/**
 * JavaScript client for the pathology inference service.
 *
 * Plain JavaScript (ES modules) with JSDoc types that `tsc --checkJs` verifies; no runtime dependencies.
 * Works in browsers and Node 20+ (global fetch, AbortController).
 *
 * @example
 *   import { PathologyClient } from "@pathology/client";
 *   const client = new PathologyClient({ baseUrl: "http://localhost:8080", apiKey: "..." });
 *   const { results } = await client.classify([{ id: "a", imageBase64 }]);
 */

/**
 * @typedef {{ id?: string, imageBase64: string }} PatchInput
 * @typedef {{ id: string, probability: number, tumor: boolean }} PatchResult
 * @typedef {{ model_version: string, threshold: number, results: PatchResult[] }} ClassifyResponse
 * @typedef {{ model_version: string, threshold: number, patch_size: number, stride: number, rows: number,
 *             cols: number, probabilities: number[], max_probability: number, windows_above_threshold: number }} HeatmapResponse
 * @typedef {{ version: string, input_name: string, output_name: string, temperature: number, threshold: number }} ModelInfo
 * @typedef {{ baseUrl: string, apiKey: string, timeoutMs?: number, retries?: number, fetch?: typeof fetch,
 *             sleep?: (ms: number) => Promise<void> }} ClientOptions
 */

/** Error raised for any non-2xx answer or a transport failure. */
export class PathologyApiError extends Error {
  /**
   * @param {string} message
   * @param {{ status?: number, code?: string, retryAfterS?: number, cause?: unknown }} [details]
   */
  constructor(message, details = {}) {
    super(message);
    this.name = "PathologyApiError";
    /** HTTP status, or 0 for a network error or timeout. */
    this.status = details.status ?? 0;
    /** Service error code: unauthorized, rate_limited, invalid_request, unavailable, internal. */
    this.code = details.code ?? "network";
    /** Seconds the service asked us to wait (429), if any. */
    this.retryAfterS = details.retryAfterS;
    this.cause = details.cause;
  }

  /** Worth retrying: rate limited, model unavailable, or no answer at all. */
  get retryable() {
    return this.status === 0 || this.status === 429 || this.status === 503;
  }
}

const MAX_PATCHES = 64;

export class PathologyClient {
  /** @param {ClientOptions} options */
  constructor(options) {
    if (!options?.baseUrl) throw new TypeError("baseUrl is required");
    if (!options.apiKey) throw new TypeError("apiKey is required");
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.retries = options.retries ?? 2;
    this.fetch = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.sleep = options.sleep ?? ((ms) => new Promise((resolve) => setTimeout(resolve, ms)));
  }

  /**
   * Classify 1-64 patches of 96x96 PNG or JPEG, base64-encoded.
   * @param {PatchInput[]} patches
   * @returns {Promise<ClassifyResponse>}
   */
  async classify(patches) {
    if (!Array.isArray(patches) || patches.length === 0 || patches.length > MAX_PATCHES) {
      throw new RangeError(`send 1 to ${MAX_PATCHES} patches`);
    }
    const body = { patches: patches.map((p) => ({ id: p.id, image_base64: p.imageBase64 })) };
    return /** @type {Promise<ClassifyResponse>} */ (this.#request("POST", "/v1/classify", body));
  }

  /**
   * Sliding-window heatmap over one tile (PNG or JPEG, base64).
   * @param {string} imageBase64
   * @param {{ stride?: number }} [options]
   * @returns {Promise<HeatmapResponse>}
   */
  async heatmap(imageBase64, options = {}) {
    const body = { image_base64: imageBase64, ...(options.stride ? { stride: options.stride } : {}) };
    return /** @type {Promise<HeatmapResponse>} */ (this.#request("POST", "/v1/heatmap", body));
  }

  /** @returns {Promise<ModelInfo>} */
  async model() {
    return /** @type {Promise<ModelInfo>} */ (this.#request("GET", "/v1/model"));
  }

  /**
   * One request with a timeout, retried with exponential backoff (or the server's Retry-After) when retryable.
   * @param {string} method
   * @param {string} path
   * @param {unknown} [body]
   * @returns {Promise<unknown>}
   */
  async #request(method, path, body) {
    /** @type {PathologyApiError | undefined} */
    let last;
    for (let attempt = 0; attempt <= this.retries; attempt += 1) {
      if (attempt > 0 && last) {
        const backoffMs = last.retryAfterS ? last.retryAfterS * 1000 : 200 * 2 ** (attempt - 1);
        await this.sleep(Math.min(backoffMs, 5_000));
      }
      try {
        return await this.#once(method, path, body);
      } catch (err) {
        if (!(err instanceof PathologyApiError) || !err.retryable) throw err;
        last = err;
      }
    }
    throw last;
  }

  /**
   * @param {string} method
   * @param {string} path
   * @param {unknown} [body]
   */
  async #once(method, path, body) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    /** @type {Response} */
    let response;
    try {
      response = await this.fetch(`${this.baseUrl}${path}`, {
        method,
        headers: { authorization: `Bearer ${this.apiKey}`, ...(body ? { "content-type": "application/json" } : {}) },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (cause) {
      const timedOut = controller.signal.aborted;
      throw new PathologyApiError(timedOut ? `timed out after ${this.timeoutMs} ms` : "network error", { cause });
    } finally {
      clearTimeout(timer);
    }
    if (response.ok) return response.json();
    /** @type {{ error?: { code?: string, message?: string } }} */
    let payload = {};
    try {
      payload = await response.json();
    } catch {
      // not JSON: keep the status only
    }
    const retryAfter = Number(response.headers.get("retry-after"));
    throw new PathologyApiError(payload.error?.message ?? `HTTP ${response.status}`, {
      status: response.status,
      code: payload.error?.code ?? "http_error",
      retryAfterS: Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : undefined,
    });
  }
}

/**
 * Probability -> RGBA colour for heatmaps (transparent below `floor`, then yellow to red).
 * @param {number} p
 * @param {number} [floor]
 * @returns {[number, number, number, number]}
 */
export function heatColor(p, floor = 0.1) {
  if (!(p >= floor)) return [0, 0, 0, 0];
  const t = Math.min(1, Math.max(0, (p - floor) / (1 - floor)));
  return [255, Math.round(220 * (1 - t)), 0, Math.round(80 + 140 * t)];
}
