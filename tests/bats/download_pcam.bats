#!/usr/bin/env bats
# scripts/download_pcam.sh against a local file server with two small files and their real MD5 sums.

setup() {
  SCRIPT="$BATS_TEST_DIRNAME/../../scripts/download_pcam.sh"
  SRV="$BATS_TEST_TMPDIR/srv"
  OUT="$BATS_TEST_TMPDIR/out"
  mkdir -p "$SRV/a.bin" "$SRV/b.csv"
  printf 'first file\n' >"$SRV/a.bin/content"
  printf 'second,file\n' >"$SRV/b.csv/content"
  export PCAM_CHECKSUMS="$BATS_TEST_TMPDIR/sums"
  printf '%s a.bin\n%s b.csv\n' "$(md5sum <"$SRV/a.bin/content" | cut -d' ' -f1)" \
    "$(md5sum <"$SRV/b.csv/content" | cut -d' ' -f1)" >"$PCAM_CHECKSUMS"
  PORT_FILE="$BATS_TEST_TMPDIR/port"
  python3 -c '
import http.server, socketserver, sys, functools
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=sys.argv[1])
handler.log_message = lambda *a: None
with socketserver.TCPServer(("127.0.0.1", 0), handler) as s:
    open(sys.argv[2], "w").write(str(s.server_address[1])); s.serve_forever()
' "$SRV" "$PORT_FILE" &
  SERVER_PID=$!
  for _ in $(seq 1 50); do [[ -s $PORT_FILE ]] && break; sleep 0.1; done
  export PCAM_BASE="http://127.0.0.1:$(cat "$PORT_FILE")" RETRIES=1
}

teardown() {
  kill "$SERVER_PID" 2>/dev/null || true
}

@test "downloads every file and verifies it" {
  run bash "$SCRIPT" "$OUT"
  [ "$status" -eq 0 ]
  [ "$(cat "$OUT/a.bin")" = "first file" ]
  [[ "$output" == *"verified b.csv"* ]]
}

@test "skips files that are already present and correct" {
  bash "$SCRIPT" "$OUT"
  run bash "$SCRIPT" "$OUT"
  [ "$status" -eq 0 ]
  [[ "$output" == *"ok       a.bin"* && "$output" == *"ok       b.csv"* ]]
}

@test "a file that fails its checksum is deleted and the run stops" {
  printf 'tampered\n' >"$SRV/b.csv/content"
  run bash "$SCRIPT" "$OUT"
  [ "$status" -eq 1 ]
  [[ "$output" == *"checksum mismatch for b.csv"* ]]
  [ ! -e "$OUT/b.csv" ]
  [ -s "$OUT/a.bin" ]
}
