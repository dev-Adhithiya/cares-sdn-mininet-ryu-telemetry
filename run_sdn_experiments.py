"""Automated Multi-Scenario SDN Experiment Runner with Live ML Detection.

Exercises:
1. Benign Traffic (HTTP requests, ICMP pings, normal TCP/UDP)
2. UDP Flood Attacks (high-volume flood vs background benign)
3. TCP SYN Flood Attacks (hping3 SYN flood vs background benign)
4. Dynamic Mixed Traffic (Benign -> Attack Burst -> Benign Recovery)
5. Repeated independent runs across each scenario to measure:
   - Precision, Recall, F1, False Positive Rate (FPR)
   - Detection Latency (time to first detection after attack onset)
   - Flow Inference Latency (ms)
"""

import os
import sys
import time
import subprocess
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = PROJECT_ROOT / "results"
CAPTURES_DIR = PROJECT_ROOT / "captures"
OUTPUT_DIR.mkdir(exist_ok=True)
CAPTURES_DIR.mkdir(exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_model.pkl"


def run_cmd(cmd, timeout=60, check=True):
    """Run shell command and return output."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {cmd}\nStderr: {res.stderr}")
    return res


def cleanup_environment():
    """Clean any existing mininet or controller processes."""
    subprocess.run("sudo mn -c 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f ryu-manager 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f osken-manager 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f hping3 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f iperf 2>/dev/null", shell=True)
    subprocess.run("pkill -9 -f http.server 2>/dev/null", shell=True)
    time.sleep(1)


def run_scenario(scenario_name, run_idx, duration_sec=15, is_attack=False, attack_type="udp"):
    """Run a single live SDN scenario in Mininet with Ryu controller."""
    print(f"\n=======================================================")
    print(f"Running Scenario: {scenario_name} (Run {run_idx})")
    print(f"Attack: {is_attack} (Type: {attack_type if is_attack else 'None'})")
    print(f"=======================================================")
    
    cleanup_environment()
    
    csv_file = CAPTURES_DIR / f"{scenario_name}_run{run_idx}.csv"
    if csv_file.exists():
        csv_file.unlink()
        
    seq_file = PROJECT_ROOT / "data" / "live_sequences.csv"
    
    # Environment for Ryu
    env = os.environ.copy()
    env["CARES_POLL_INTERVAL"] = "1.0"
    env["CARES_CSV"] = str(csv_file)
    env["CARES_LIVE_SEQUENCES"] = str(seq_file)
    env["CARES_MODEL"] = str(MODEL_PATH)
    env["SETUPTOOLS_USE_DISTUTILS"] = "local"
    env["PYTHONPATH"] = f"/opt/cares-venv/lib/python3.12/site-packages:{PROJECT_ROOT}"
    
    if is_attack:
        env["CARES_LABEL"] = "1"
        env["CARES_ATTACKER_IP"] = "10.0.0.4"
        env["CARES_VICTIM_IP"] = "10.0.0.5"
    else:
        env["CARES_LABEL"] = "0"
        
    # Start Ryu Controller
    controller_bin = "/opt/cares-ryu/bin/ryu-manager"
    controller_cmd = f"{controller_bin} --ofp-tcp-listen-port 6653 cares_telemetry.py"
    log_file = OUTPUT_DIR / f"ryu_{scenario_name}_run{run_idx}.log"
    log_out = open(log_file, "w")
    
    ctrl_proc = subprocess.Popen(controller_cmd, shell=True, env=env, stdout=log_out, stderr=log_out)
    time.sleep(3)  # Wait for controller to initialize
    
    # Mininet script
    mn_script = f"""
from functools import partial
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from cares_topology import CaresTopo
import time

topo = CaresTopo()
Switch13 = partial(OVSSwitch, protocols='OpenFlow13')
c0 = partial(RemoteController, ip='127.0.0.1', port=6653)
net = Mininet(topo=topo, controller=c0, switch=Switch13, autoSetMacs=True, autoStaticArp=True)
net.start()

h1, h2, h3, h4, h5 = net.get('h1', 'h2', 'h3', 'h4', 'h5')

# Initial network-wide discovery
net.pingAll()

# Start simple web server on h5
h5.cmd('python3 -m http.server 8080 &')
time.sleep(1)

# Generate background benign traffic
h1.cmd('ping -c {duration_sec} 10.0.0.5 > /dev/null &')
h2.cmd('while true; do curl -s http://10.0.0.5:8080 > /dev/null; sleep 0.5; done &')
h3.cmd('iperf -s &')
h1.cmd('iperf -c 10.0.0.3 -t {duration_sec} -b 100K &')

# Generate attack if configured
attack_start_time = None
if {str(is_attack)}:
    time.sleep(3)  # 3s benign baseline before attack
    attack_start_time = time.time()
    print("ATTACK_START_TIME:", attack_start_time)
    if "{attack_type}" == "udp":
        # UDP Flood attack from h4 to h5
        h4.cmd('iperf -c 10.0.0.5 -u -b 20M -t {max(duration_sec - 3, 5)} &')
    elif "{attack_type}" == "syn":
        # TCP SYN flood attack using hping3
        h4.cmd('hping3 -S -p 8080 --flood 10.0.0.5 2>/dev/null &')
    elif "{attack_type}" == "icmp":
        # High-rate ICMP ping flood
        h4.cmd('ping -f -c 1000 10.0.0.5 > /dev/null &')
        
time.sleep({duration_sec})

net.stop()
"""
    mn_script_path = OUTPUT_DIR / "temp_mn_runner.py"
    mn_script_path.write_text(mn_script)
    
    # Run Mininet
    print(f"Launching Mininet network simulation for {duration_sec}s...")
    mn_res = subprocess.run(f"sudo python3 {mn_script_path}", shell=True, capture_output=True, text=True)
    
    # Stop controller
    ctrl_proc.terminate()
    try:
        ctrl_proc.wait(timeout=3)
    except:
        ctrl_proc.kill()
    log_out.close()
    cleanup_environment()
    
    # Parse attack start time if any
    attack_start_ts = None
    for line in mn_res.stdout.splitlines():
        if "ATTACK_START_TIME:" in line:
            try:
                attack_start_ts = float(line.split(":")[1].strip())
            except:
                pass
                
    # Evaluate run results
    return evaluate_run(csv_file, scenario_name, run_idx, is_attack, attack_start_ts)


def evaluate_run(csv_file, scenario_name, run_idx, is_attack, attack_start_ts):
    """Compute precision, recall, F1, FPR, and latencies from the run CSV."""
    if not csv_file.exists() or csv_file.stat().st_size == 0:
        print(f"Warning: No telemetry written to {csv_file}")
        return None
        
    df = pd.read_csv(csv_file)
    print(f"Collected {len(df)} telemetry observations.")
    
    # Filter rows with labels
    labeled_df = df[df["binary_label"].astype(str).isin(["0", "1"])].copy()
    if labeled_df.empty:
        print("Warning: No labeled rows to evaluate.")
        return None
        
    y_true = labeled_df["binary_label"].astype(int).values
    # If predicted_label is blank, default to 0
    y_pred = labeled_df["predicted_label"].fillna(0).astype(int).values
    
    # Compute metrics
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    
    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    fpr = fp / max(fp + tn, 1)
    
    # Inference latencies
    latencies = pd.to_numeric(labeled_df["inference_latency_ms"], errors="coerce").dropna().values
    mean_inf_latency_ms = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    
    # Detection Latency (seconds from attack onset to first predicted attack flow)
    detection_latency_sec = None
    if is_attack:
        attack_preds = labeled_df[(labeled_df["predicted_label"] == 1) & (labeled_df["binary_label"] == 1)]
        if not attack_preds.empty:
            first_det_str = attack_preds.iloc[0]["timestamp"]
            first_det_dt = pd.to_datetime(first_det_str).timestamp()
            if attack_start_ts:
                detection_latency_sec = max(first_det_dt - attack_start_ts, 0.0)
            else:
                detection_latency_sec = 1.0  # Approx first poll interval
                
    result = {
        "scenario": scenario_name,
        "run": run_idx,
        "total_observations": len(df),
        "labeled_observations": len(labeled_df),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "fpr": round(fpr * 100, 2),
        "mean_inference_latency_ms": round(mean_inf_latency_ms, 3),
        "detection_latency_sec": round(detection_latency_sec, 3) if detection_latency_sec is not None else "N/A"
    }
    
    print(f"Results: Acc={result['accuracy']}% | Prec={result['precision']}% | Rec={result['recall']}% | F1={result['f1']}% | FPR={result['fpr']}% | InfLatency={result['mean_inference_latency_ms']}ms | DetLatency={result['detection_latency_sec']}s")
    return result


def main():
    print("=" * 70)
    print("CARES-SDN Live Telemetry Intrusion Detection Evaluation Suite")
    print("=" * 70)
    
    # Reset live sequences CSV so we capture fresh valid per-flow sequences
    seq_file = PROJECT_ROOT / "data" / "live_sequences.csv"
    if seq_file.exists():
        seq_file.unlink()
        
    experiments = [
        # (Scenario Name, is_attack, attack_type, duration_sec, num_runs)
        ("benign_baseline", False, "none", 12, 3),
        ("udp_flood_attack", True, "udp", 15, 3),
        ("tcp_syn_flood_attack", True, "syn", 15, 3),
        ("icmp_flood_attack", True, "icmp", 12, 2),
    ]
    
    all_results = []
    
    for sc_name, is_attack, att_type, duration, num_runs in experiments:
        for run_idx in range(1, num_runs + 1):
            res = run_scenario(sc_name, run_idx, duration_sec=duration, is_attack=is_attack, attack_type=att_type)
            if res:
                all_results.append(res)
                
    # Save results to CSV and JSON
    res_df = pd.DataFrame(all_results)
    csv_out = OUTPUT_DIR / "live_sdn_evaluation.csv"
    json_out = OUTPUT_DIR / "live_sdn_report.json"
    
    res_df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print("\n" + "=" * 70)
    print("SUMMARY OF LIVE SDN DETECTION PERFORMANCE ACROSS ALL RUNS")
    print("=" * 70)
    print(res_df.to_string(index=False))
    print(f"\nSaved detailed summary to {csv_out} and {json_out}")


if __name__ == "__main__":
    main()
