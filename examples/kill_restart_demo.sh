#!/usr/bin/env bash
# Repeatable force-kill / restart demo against the docker compose stack.
#
# What it shows, end to end:
#   1. a model is serving (and a slow inference is in flight);
#   2. the API process alone is SIGKILLed -- the model worker survives as an
#      orphan (this is the failure the duty engineer hit);
#   3. the in-container supervisor restarts the API in the same container;
#   4. startup recovery terminates the orphaned worker *before* the API opens
#      for traffic, and /admin/status reports exactly what was reclaimed;
#   5. the old request is not replayed; a model is re-uploaded and inference
#      works again on the same data directory.
#
# Usage:  examples/kill_restart_demo.sh            (uses docker compose)
#         BASE=http://host:port examples/kill_restart_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE:-http://127.0.0.1:8000}"
SERVICE="${COMPOSE_SERVICE:-qc-inference}"
RUN_ID="${RUN_ID:-$(date +%s)}"
V1="rc-v1-$RUN_ID"

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
jqt()  { python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), indent=2)[:900])'; }

wait_healthz() {
  for _ in $(seq 1 90); do
    if curl -fsS "$BASE/healthz" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  echo "API did not become healthy in time" >&2
  return 1
}

model_json() { # version
  cat <<JSON
{"version": "$1",
 "weights": [[1.2,-0.4,0.0,0.3],[-0.2,0.9,0.5,-0.1],[0.1,-0.8,1.1,0.2]],
 "bias": [-0.1,0.05,-0.2],
 "labels": ["pass","scratch","dent"]}
JSON
}

say "0. build and start the stack"
docker compose up -d --build
wait_healthz
info "API healthy at $BASE (Swagger UI: $BASE/docs)"

say "1. upload model $V1 and run inference"
curl -fsS -X POST "$BASE/admin/models" -H 'content-type: application/json' \
  --data-binary "$(model_json "$V1")" | jqt
curl -fsS -X POST "$BASE/infer" -H 'content-type: application/json' \
  -d '{"features":[0.7,0.2,0.1,0.5]}' | jqt
WORKER_PID=$(curl -fsS "$BASE/admin/status" | python3 -c \
  'import json,sys; st=json.load(sys.stdin); print(next(s["pid"] for s in st["slots"] if s["process_state"]=="active"))')
info "active model worker pid (container pid namespace): $WORKER_PID"

say "2. start a 30s inference so the kill lands mid-flight"
curl -sS -X POST "$BASE/infer" -H 'content-type: application/json' \
  -d '{"features":[0.7,0.2,0.1,0.5],"slow_ms":30000,"timeout_ms":60000}' >/dev/null 2>&1 &
SLOW_CURL=$!
sleep 1
info "slow request is running inside worker $WORKER_PID"

say "3. SIGKILL the API process only (model worker survives, orphaned)"
docker compose exec -T "$SERVICE" sh -c "
  kill -9 \"\$(cat /data/api.pid)\"
  sleep 0.3
  if [ -d /proc/$WORKER_PID ]; then
    echo \"   [container] API is dead; worker $WORKER_PID is STILL RUNNING as an orphan\"
  else
    echo \"   [container] worker $WORKER_PID already gone (unexpected)\"
  fi"
kill "$SLOW_CURL" 2>/dev/null || true  # the in-flight client died with the API

say "4. supervisor restarts the API; startup recovery reclaims the orphan"
wait_healthz
curl -fsS "$BASE/admin/status" | jqt
curl -fsS "$BASE/admin/status" | python3 -c "
import json, sys
st = json.load(sys.stdin)
rec = st.get('recovery') or {}
pids = [w.get('pid') for w in rec.get('reclaimed_workers', [])]
assert $WORKER_PID in pids, f'worker $WORKER_PID missing from reclaimed list: {pids}'
assert st['slots_in_use'] == 0, st['slots']
assert st['total_accepted'] == 0, 'old in-flight request must not be replayed'
print(f'   OK: recovery reclaimed orphaned worker pid $WORKER_PID; all slots idle; nothing replayed')
"

say "5. re-upload the SAME version id and infer on the recovered service"
info "the registry is in-memory, so $V1 can be uploaded again after recovery"
curl -fsS -X POST "$BASE/admin/models" -H 'content-type: application/json' \
  --data-binary "$(model_json "$V1")" | jqt
curl -fsS -X POST "$BASE/infer" -H 'content-type: application/json' \
  -d '{"features":[0.7,0.2,0.1,0.5]}' | jqt
curl -fsS "$BASE/admin/status" | jqt

say "done: SIGKILL -> orphan reclaimed at startup -> serving again on the same data dir"
info "re-run any time: examples/kill_restart_demo.sh (fresh version ids each run)"
