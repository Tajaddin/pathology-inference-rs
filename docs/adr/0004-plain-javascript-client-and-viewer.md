# 0004. Plain JavaScript for the client library and the viewer

Date: 2026-10-18. Status: accepted.

## Context

The front end is small (one viewer page) and the client library should drop into any JavaScript codebase,
browser or Node, without a build step or runtime dependencies.

## Decision

- ES modules in plain JavaScript, typed with JSDoc and checked by `tsc --checkJs` in CI, linted by ESLint.
- The client has no dependencies: global `fetch`, `AbortController` timeouts, retries with exponential backoff
  that honour `Retry-After`, and one error type carrying the service's error code.
- The viewer draws the tile and the heatmap on two canvases; Vite builds it; nginx (unprivileged) serves it.
- Node's built-in test runner tests the client with a fake `fetch`; Playwright runs the viewer against the real
  Rust service in CI.

## Consequences

- No framework to learn or upgrade; a larger UI would justify one.
- The demo API key for the local viewer is a documented placeholder; a deployed viewer would get a short-lived
  token from its host rather than a static key.
