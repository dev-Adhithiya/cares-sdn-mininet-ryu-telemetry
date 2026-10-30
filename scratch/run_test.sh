#!/bin/bash
set -e
sudo mn -c
pkill -9 -f ryu-manager || true
export CARES_POLL_INTERVAL=1.0
export CARES_CSV="/tmp/test_telemetry.csv"
export CARES_MODEL="/opt/cares-project/models/live_xgboost.pkl"
/opt/cares-ryu/bin/ryu-manager --ofp-tcp-listen-port 6653 /opt/cares-project/cares_telemetry.py > /tmp/ryu_test.log 2>&1 &
RYU_PID=$!
sleep 4
sudo /opt/cares-venv/bin/python3 /opt/cares-project/scratch/test_mininet_ryu.py
sleep 2
kill $RYU_PID || true
echo "=== RYU LOG ==="
cat /tmp/ryu_test.log
echo "=== CSV ROWS ==="
wc -l /tmp/test_telemetry.csv 2>/dev/null || echo "No CSV"
