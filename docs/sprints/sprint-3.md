# Sprint 3 retro: JavaScript viewer and client

Dates: 2026-10-18. Milestone: [Sprint 3 - JavaScript viewer and client](https://github.com/Tajaddin/pathology-inference-rs/milestone/3).

**Goal:** a plain-JavaScript slide viewer that shows tumour heatmaps from the service, and a JavaScript client
library with tests. **Met.**

## Delivered

| Issue | Points | Result |
|---|---|---|
| #11 client library | 3 | ES modules, no dependencies, JSDoc types checked by `tsc`; retries with `Retry-After`, timeouts, typed errors |
| #12 viewer | 5 | demo tiles of real test patches, heatmap overlay at stride 96 or 48, ground-truth outlines, round-trip time |

8 of 8 points done.

## Measured

- Client: 8 `node:test` tests, 98.8% line coverage.
- Viewer bundle: 5.7 kB of JavaScript (2.7 kB gzipped).
- Playwright against the real Rust service with the trained model: 4 of 4 tests pass (desktop and mobile).
  Scoring the mixed tile at stride 48 (225 windows) took a 219 ms round trip in the screenshot run (one run, local
  stack, default service settings).

## What went well

- The end-to-end test drives the real stack (Rust service, ONNX model, viewer) rather than a mock, so it checks
  CORS, authentication and the heatmap layout together.

## What did not

- `npm test` had never run: `node --test client/test/` treats a directory as a file on Node 24. It now passes a glob.
- `tsc` failed twice: `@types/node` was missing, and with `checkJs` it type-checked a JavaScript file inside
  `node_modules`; `maxNodeModuleJsDepth: 0` stops that.
- The viewer said its probabilities were calibrated; on the test split they are not (Sprint 1). The legend now
  says to read them as a ranking.

## Carried into Sprint 4

- Nothing open.
