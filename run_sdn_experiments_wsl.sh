#!/usr/bin/env bash
# run_sdn_experiments_wsl.sh
# ============================================================
# Runs the full CARES-SDN automated evaluation suite inside WSL.
# Handles PYTHONPATH, sudo, OVS, and Ryu controller lifecycle.
#
# Usage (from Windows PowerShell):
#   wsl -d cares-ubuntu bash /mnt/e/CN\'/run_sdn_experiments_wsl.sh
#
# Usage (from WSL shell):
#   bash /mnt/e/CN\'/run_sdn_experiments_wsl.sh
# ============================================================

set -euo pipefail

PROJECT_ROOT="/opt/cares-project"
VENV="/opt/cares-venv"
RYU_VENV="/opt/cares-ryu"
PYTHON="${VENV}/bin/python3"
RYU_BIN="${RYU_VENV}/bin/ryu-manager"
RESULTS="${PROJECT_ROOT}/results"
CAPTURES="${PROJECT_ROOT}/captures"
LOG_DIR="${RESULTS}"

mkdir -p "${RESULTS}" "${CAPTURES}"

export PYTHONPATH="${VENV}/lib/python3.12/site-packages:${PROJECT_ROOT}"
export SETUPTOOLS_USE_DISTUTILS=local

echo "================================================"
echo " CARES-SDN Experiment Runner (WSL)"
echo " Project : ${PROJECT_ROOT}"
echo " Venv    : ${VENV}"
echo "================================================"

# ── ensure OVS is running ────────────────────────────────────────────────────
if ! pgrep ovs-vswitchd &>/dev/null; then
    echo "[setup] Starting Open vSwitch..."
    sudo service openvswitch-switch start || true
    sleep 2
fi

# ── cleanup helper ────────────────────────────────────────────────────────────
cleanup() {
    sudo mn -c &>/dev/null || true
    pkill -9 -f ryu-manager  &>/dev/null || true
    pkill -9 -f osken-manager &>/dev/null || true
    pkill -9 -f hping3        &>/dev/null || true
    pkill -9 -f iperf         &>/dev/null || true
    sleep 1
}

# ── run one scenario ─────────────────────────────────────────────────────────
run_scenario() {
    local SCENARIO_NAME="$1"
    local RUN_IDX="$2"
    local DURATION="$3"
    local IS_ATTACK="$4"   # "true" or "false"
    local ATTACK_TYPE="$5" # udp / syn / icmp / none

    echo ""
    echo "─── Scenario: ${SCENARIO_NAME}  Run: ${RUN_IDX}  Attack: ${IS_ATTACK} (${ATTACK_TYPE}) ───"

    cleanup

    local CSV_FILE="${CAPTURES}/${SCENARIO_NAME}_run${RUN_IDX}.csv"
    local SEQ_FILE="${PROJECT_ROOT}/data/live_sequences.csv"
    local LOG_FILE="${LOG_DIR}/ryu_${SCENARIO_NAME}_run${RUN_IDX}.log"
    local MODEL_PATH="${PROJECT_ROOT}/models/live_xgboost.pkl"
    [ -f "${MODEL_PATH}" ] || MODEL_PATH="${PROJECT_ROOT}/models/live_random_forest.pkl"
    [ -f "${MODEL_PATH}" ] || MODEL_PATH="${PROJECT_ROOT}/models/random_forest_model.pkl"

    rm -f "${CSV_FILE}"

    # ── environment ──────────────────────────────────────────────────────────
    export CARES_POLL_INTERVAL=1.0
    export CARES_CSV="${CSV_FILE}"
    export CARES_LIVE_SEQUENCES="${SEQ_FILE}"
    export CARES_MODEL="${MODEL_PATH}"
    export CARES_SCALER="${PROJECT_ROOT}/models/live_scaler.pkl"

    if [ "${IS_ATTACK}" = "true" ]; then
        export CARES_LABEL=1
        export CARES_ATTACKER_IP=10.0.0.4
        export CARES_VICTIM_IP=10.0.0.5
    else
        export CARES_LABEL=0
        unset CARES_ATTACKER_IP CARES_VICTIM_IP 2>/dev/null || true
    fi

    # ── start Ryu ────────────────────────────────────────────────────────────
    "${RYU_BIN}" --ofp-tcp-listen-port 6653 \
        "${PROJECT_ROOT}/cares_telemetry.py" \
        >"${LOG_FILE}" 2>&1 &
    RYU_PID=$!
    sleep 4   # wait for controller to be ready

    # ── Mininet scenario ──────────────────────────────────────────────────────
    MN_SCRIPT=$(mktemp /tmp/mn_XXXXXX.py)
    cat > "${MN_SCRIPT}" << PYEOF
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/usr/lib/python3/dist-packages')
sys.path.insert(0, '${VENV}/lib/python3.12/site-packages')
sys.path.insert(0, '${PROJECT_ROOT}')

from functools import partial
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel
from cares_topology import CaresTopo
import time

setLogLevel('warning')

topo  = CaresTopo()
SW13  = partial(OVSSwitch, protocols='OpenFlow13')
RC    = partial(RemoteController, ip='127.0.0.1', port=6653)
net   = Mininet(topo=topo, controller=RC, switch=SW13,
                autoSetMacs=True, autoStaticArp=True)
net.start()

h1, h2, h3, h4, h5 = net.get('h1','h2','h3','h4','h5')

# ── network discovery ────────────────────────────────────────────────────────
net.pingAll()

# ── benign background traffic ────────────────────────────────────────────────
h5.cmd('python3 -m http.server 8080 &')
time.sleep(1)
h1.cmd('ping -c ${DURATION} 10.0.0.5 > /dev/null &')
h2.cmd('while true; do wget -q -O /dev/null http://10.0.0.5:8080/ 2>/dev/null; sleep 0.5; done &')
h3.cmd('iperf -s &')
h1.cmd('iperf -c 10.0.0.3 -t ${DURATION} -b 100K &')

# ── attack traffic ───────────────────────────────────────────────────────────
ATTACK_DURATION = max(${DURATION} - 3, 5)
if '${IS_ATTACK}' == 'true':
    time.sleep(3)
    attack_start = time.time()
    print("ATTACK_START_TIME:", attack_start, flush=True)
    if '${ATTACK_TYPE}' == 'udp':
        h4.cmd(f'iperf -c 10.0.0.5 -u -b 20M -t {ATTACK_DURATION} &')
    elif '${ATTACK_TYPE}' == 'syn':
        h4.cmd('hping3 -S -p 8080 --flood 10.0.0.5 2>/dev/null &')
    elif '${ATTACK_TYPE}' == 'icmp':
        h4.cmd(f'ping -f -c 5000 10.0.0.5 > /dev/null &')

time.sleep(${DURATION})
net.stop()
PYEOF

    echo "[mn ] Launching Mininet for ${DURATION}s..."
    sudo "${PYTHON}" "${MN_SCRIPT}" 2>&1
    MN_STATUS=$?
    rm -f "${MN_SCRIPT}"

    # ── stop Ryu ─────────────────────────────────────────────────────────────
    kill "${RYU_PID}" 2>/dev/null || true
    wait "${RYU_PID}" 2>/dev/null || true
    cleanup

    if [ ${MN_STATUS} -ne 0 ]; then
        echo "[WARN] Mininet exited with status ${MN_STATUS}"
    fi

    # ── report rows ───────────────────────────────────────────────────────────
    if [ -f "${CSV_FILE}" ]; then
        local ROWS
        ROWS=$(wc -l < "${CSV_FILE}")
        echo "[ok ] CSV written: ${CSV_FILE}  (${ROWS} lines)"
    else
        echo "[WARN] No CSV written for ${SCENARIO_NAME} run ${RUN_IDX}"
    fi
}

# ── reset live sequences ──────────────────────────────────────────────────────
rm -f "${PROJECT_ROOT}/data/live_sequences.csv"

# ── experiment schedule ───────────────────────────────────────────────────────
#  (scenario_name   run_idx  duration  is_attack  attack_type)
run_scenario "benign_baseline"      1  12  false  none
run_scenario "benign_baseline"      2  12  false  none
run_scenario "benign_baseline"      3  12  false  none
run_scenario "udp_flood_attack"     1  15  true   udp
run_scenario "udp_flood_attack"     2  15  true   udp
run_scenario "udp_flood_attack"     3  15  true   udp
run_scenario "tcp_syn_flood_attack" 1  15  true   syn
run_scenario "tcp_syn_flood_attack" 2  15  true   syn
run_scenario "tcp_syn_flood_attack" 3  15  true   syn
run_scenario "icmp_flood_attack"    1  12  true   icmp
run_scenario "icmp_flood_attack"    2  12  true   icmp

echo ""
echo "================================================"
echo " All scenarios complete. Running evaluation..."
echo "================================================"
"${PYTHON}" "${PROJECT_ROOT}/ml_scripts/evaluate_live_captures.py"

echo ""
echo "================================================"
echo " Training LSTM on collected sequences..."
echo "================================================"
if [ -f "${PROJECT_ROOT}/data/live_sequences.csv" ]; then
    NROWS=$(wc -l < "${PROJECT_ROOT}/data/live_sequences.csv")
    echo "live_sequences.csv has ${NROWS} rows"
fi
# Always train LSTM on CICIOT23 contract data (reliable baseline)
"${PYTHON}" "${PROJECT_ROOT}/ml_scripts/train_lstm_ciciot.py"

echo ""
echo "Done. Results in ${RESULTS}/"
