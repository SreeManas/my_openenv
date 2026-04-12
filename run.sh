#!/bin/bash
# run.sh — Start CodeReviewBench server, then optionally run inference
#
# Usage:
#   ./run.sh            → server only (port 7860)
#   ./run.sh inference  → server + inference agent

set -e
PORT=7860

# Kill any existing process on the port before starting
if lsof -ti:$PORT &>/dev/null; then
    echo "Port $PORT already in use — killing existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 1
fi

SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ""
        echo "Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting CodeReviewBench server on port $PORT..."
python3.11 -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --log-level warning &
SERVER_PID=$!

# Wait up to 10s for server to be ready
for i in {1..20}; do
    if curl -sf "http://localhost:$PORT/tasks" > /dev/null 2>&1; then
        TASK_COUNT=$(curl -s "http://localhost:$PORT/tasks" | python3.11 -c "import sys,json; print(len(json.load(sys.stdin)))")
        echo "Server ready: $TASK_COUNT tasks available"
        break
    fi
    sleep 0.5
done

if [ "$1" = "inference" ]; then
    echo ""
    echo "Running inference agent..."
    echo ""
    ENV_URL=http://localhost:$PORT python3.11 inference.py
else
    echo "Server running at http://localhost:$PORT"
    echo "  Run agent : python3.11 inference.py"
    echo "  Stop      : Ctrl+C"
    wait "$SERVER_PID"
fi
