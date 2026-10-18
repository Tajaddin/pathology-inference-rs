// k6 load test for POST /v1/classify at a fixed arrival rate (JavaScript).
//
//   docker run --rm --network pathology_default -v "$PWD/bench/k6:/scripts:ro" \
//     -v "$PWD/service/tests/fixtures:/fixtures:ro" -v "$PWD/reports/k6:/out" \
//     -e FIXTURES_DIR=/fixtures -e TARGET=http://service:8080 -e LABEL=rust -e RATE=200 -e PATCHES=1 \
//     grafana/k6:1.3.0 run /scripts/classify.js
//
// Each request carries PATCHES real PCam test patches (base64 PNG): the parity fixtures in service/tests/fixtures.
import http from "k6/http";
import encoding from "k6/encoding";
import { check } from "k6";
import { SharedArray } from "k6/data";

const FIXTURES_DIR = __ENV.FIXTURES_DIR || "../../service/tests/fixtures";
const fixtures = new SharedArray("patches", () => {
  const meta = JSON.parse(open(`${FIXTURES_DIR}/parity.json`));
  return meta.map((m) => encoding.b64encode(open(`${FIXTURES_DIR}/${m.file}`, "b")));
});

const RATE = Number(__ENV.RATE || 100);
const PATCHES = Number(__ENV.PATCHES || 1);
const LABEL = __ENV.LABEL || "rust";
const TARGET = __ENV.TARGET || "http://localhost:8080";
const KEY = __ENV.API_KEY || "local-dev-key-loadtest-0001";

export const options = {
  discardResponseBodies: true,
  scenarios: {
    steady: {
      executor: "constant-arrival-rate",
      rate: RATE,
      timeUnit: "1s",
      duration: __ENV.DURATION || "60s",
      preAllocatedVUs: Math.max(20, RATE / 2),
      maxVUs: Math.max(200, RATE * 2),
    },
  },
  summaryTrendStats: ["avg", "med", "p(90)", "p(95)", "p(99)", "max"],
};

export default function () {
  const patches = [];
  for (let i = 0; i < PATCHES; i += 1) {
    patches.push({ id: String(i), image_base64: fixtures[(__ITER + i) % fixtures.length] });
  }
  const res = http.post(`${TARGET}/v1/classify`, JSON.stringify({ patches }), {
    headers: { "content-type": "application/json", authorization: `Bearer ${KEY}` },
  });
  check(res, { "status 200": (r) => r.status === 200 });
}

export function handleSummary(data) {
  const d = data.metrics.http_req_duration.values;
  const reqs = data.metrics.http_reqs.values;
  const failed = data.metrics.http_req_failed.values.rate;
  const line = `${LABEL} ${RATE}/s x ${PATCHES} patches: ${reqs.count} requests (${reqs.rate.toFixed(1)}/s), ` +
    `failed ${(100 * failed).toFixed(2)}%, p50 ${d.med.toFixed(1)} ms, p95 ${d["p(95)"].toFixed(1)} ms, p99 ${d["p(99)"].toFixed(1)} ms\n`;
  return { [`/out/${LABEL}-r${RATE}-p${PATCHES}.json`]: JSON.stringify(data, null, 2), stdout: line };
}
