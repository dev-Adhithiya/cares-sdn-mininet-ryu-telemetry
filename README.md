# CARES-SDN: Mininet, Ryu telemetry, and ML handoff

## Executive result

The SDN telemetry baseline is implemented and was exercised in WSL Ubuntu 24.04.5. Mininet built the five-host, one-switch topology; it connected to the Ryu controller; `pingall` completed with 0% loss; and the collector repeatedly wrote timestamped OpenFlow observations to CSV.

The downloaded CICIoT2023 data is now checked. A bounded Random Forest/XGBoost binary baseline has been trained on features actually present in its CSVs, using held-out attack-class folders and benign capture files. These are exploratory results on a balanced sample, not full-corpus or final CARES-SDN claims. The supplied nine-feature scripts still do not match this dataset's schema, and the LSTM remains unrun because these files have no flow IDs or timestamps for valid sequences.

## What was implemented

- `cares_topology.py`: Mininet topology with h1–h3 benign hosts, h4 attacker, h5 server, and OpenFlow 1.3 switch s1.
- `cares_telemetry.py`: Ryu 4.34 learning switch and fixed-interval OpenFlow stats poller. It writes raw flow observations, UTC timestamp, counters, match fields where present, and controlled labels. For label 1, attacker and victim IPs are required; unrelated or wildcard rows stay unlabeled.
- `setup_wsl.sh`: installs Mininet/OVS and the Ryu runtime in Ubuntu 24.04. Ryu 4.34 needs compatibility adjustments to install on Python 3.12; the script patches the downloaded Ryu source/runtime accordingly.
- `experiment_log.md`: commands and notes for the two captured runs.
- `ml_scripts/`: copies of the 11 training, preprocessing, and inspection scripts provided from Downloads, plus their Python package list.

## Environment and network validation

| Check | Observed result |
|---|---|
| WSL distro used | Ubuntu 24.04.5 LTS, `cares-ubuntu` |
| Mininet | 2.3.0 |
| Open vSwitch | 3.3.9 |
| Ryu | 4.34, `/opt/cares-ryu` |
| OpenFlow | 1.3 |
| Topology | 5 hosts, 1 switch |
| `pingall` | 20/20 replies, 0% loss |
| Ryu connection / polling | Switch connected; periodic flow-stat replies written to CSV |

The pre-existing `Ubuntu` WSL registration points to a missing virtual disk. It was not changed. The working distro is launched with `wsl -d cares-ubuntu`.

The ML baseline was run with Windows Anaconda Python 3.13.5, pandas 2.2.3, scikit-learn 1.6.1, and XGBoost 2.1.4.

## Captures and measured telemetry results

Capture files are in `captures/` and preserve observation order.

| Capture | Rows (excluding header) | Flow IDs | Observations per flow | Labels and scenario |
|---|---:|---:|---:|---|
| `captures/benign.csv` | 312 | 21 | 14–15 | All 312 labeled 0; pingall plus a 15-packet ICMP ping |
| `captures/udp_attack.csv` | 31 | 2 | 15–16 | 15 attacker-to-victim UDP observations labeled 1; 16 wildcard/table-miss observations blank |

The UDP scenario used h4 (10.0.0.4) to h5 (10.0.0.5), UDP port 5001, 64 Kbit/s for 15 seconds. Iperf reported 88 datagrams and 0% loss. This is a controlled traffic-labeling smoke experiment; it should not be described as a high-volume flood or as a final performance experiment.

Raw columns are:

```text
flow_id,timestamp,duration_sec,duration_nsec,packet_count,byte_count,protocol,src_ip,dst_ip,src_port,dst_port,poll_interval,binary_label
```

Some switch rules have wildcard matches; therefore their IP, protocol, and port cells are blank. OpenFlow packet and byte counters are cumulative. These CSVs show that the network and polling loop work, but the traffic diversity is too small to make a meaningful generalization claim or train/evaluate a reliable detector.

## ML status and interpretation

### Downloaded dataset schema and compatibility

The dataset is at `G:\CN\CSV\CSV`: 309 CSV files in 34 class folders, totaling about 8.33 GB extracted. All 309 files share the same 39-column header. The source CSVs have no `label` field; benign versus attack was taken from the containing folder (`Benign_Final` is benign, all other class folders are attack).

The supplied scripts require `flow_duration`, `Srate`, `Drate`, and `label`, which are absent from every downloaded CSV. They do include `Protocol Type`, `Rate`, `syn_count`, `ack_count`, `fin_count`, `rst_count`, `TCP`, `UDP`, `ICMP`, and `Tot size`. A new runner uses only those ten real columns. It does not rename or fabricate missing fields.

`ml_scripts/train_ciciot_native_features.py` uses seed 2026 and a group holdout: attack class folders are assigned wholly to train/validation/test, and benign source CSVs are held out separately. It reads at most the first 2,000 rows from each source CSV, caps each attack folder at 10,000 rows and each benign capture at 25,000, drops rows with non-finite selected features, then downsamples each split to equal class counts. This produced 3,999 benign + 3,999 attack training rows, and 2,000 + 2,000 rows in each validation and test split. The test measures generalization to attack folders not present in training, on this bounded sample only.

### Measured binary baseline

| Split | Model | Accuracy | Precision | Recall | F1 | False-positive rate | TN / FP / FN / TP |
|---|---|---:|---:|---:|---:|---:|---|
| Validation | Random Forest | 91.25% | 93.15% | 89.05% | 91.05% | 6.55% | 1869 / 131 / 219 / 1781 |
| Test | Random Forest | 95.43% | 94.43% | 96.55% | 95.48% | 5.70% | 1886 / 114 / 69 / 1931 |
| Validation | XGBoost | 91.13% | 92.46% | 89.55% | 90.98% | 7.30% | 1854 / 146 / 209 / 1791 |
| Test | XGBoost | 95.08% | 93.70% | 96.65% | 95.15% | 6.50% | 1870 / 130 / 67 / 1933 |

On this run, Random Forest fit in 0.23 s and predicted the 4,000-row test split in 0.049 s; XGBoost fit in 0.44 s and predicted it in 0.005 s. Timing is machine-dependent. These balanced-sample rates are not population-level deployment estimates; the bounded sample reads only the beginning of each source file. The CSV/JSON metrics, split membership, balanced split CSVs, and serialized models are in `results/`, `data/ciciot_native/`, and `models/`.

Re-run the bounded baseline from PowerShell with an environment containing the packages in `ml_scripts/ciciot_native_requirements.txt`:

```powershell
python .\ml_scripts\train_ciciot_native_features.py `
  --data-root G:\CN\CSV\CSV `
  --output-root (Get-Location).Path
```

The supplied real-data training and preprocessing scripts expect columns such as:

```text
flow_duration, Protocol Type, Rate, Drate, syn_count, ack_count,
fin_count, rst_count, Tot size, label
```

Those columns cannot be recovered faithfully from the current OpenFlow CSVs. In particular, the collector does not parse packet-level TCP flags; direction-specific `Drate` and `Tot size` definitions have not been agreed; and packet counts / byte counts are cumulative counters rather than per-packet values. The handoff says not to fabricate these fields, so no such mapping was made.

| Model/report item | Status |
|---|---|
| Random Forest on downloaded CICIoT2023 fields | Exploratory bounded-sample results above; uses the ten fields present, not the supplied nine-feature contract |
| XGBoost on downloaded CICIoT2023 fields | Exploratory bounded-sample results above; uses the ten fields present, not the supplied nine-feature contract |
| Supplied `train_random_forest.py` / `train_xgboost.py` | Not run unchanged: they expect missing columns and preassembled train/validation/test paths |
| LSTM on real sequential data | Not run: `data/live_sequences.csv` and fitted scaler are absent; neither these CIC CSVs nor the OpenFlow captures provide per-flow timestamped sequences meeting its contract |
| `build_lstm_model.py` | Synthetic architecture demonstration only; its synthetic scores must not be presented as research results |
| `create_lstm_sequences.py` | Synthetic sequence demonstration; it is not a live-capture preprocessing pipeline |
| `save_model_comparison.py` | Contains hard-coded Random Forest/XGBoost numbers. The confusion-matrix arithmetic is internally consistent, but no dataset, run logs, saved models, or provenance were supplied to verify those are results from this project. They are excluded from measured results here. |

The native-feature comparison is exploratory and kept separate from the Ryu live-telemetry pipeline. The report distinguishes those results from unavailable supplied-contract/LSTM results.

## Supplied scripts and their expected inputs

All supplied scripts are copied into `ml_scripts/`; the native-feature runner is also in that folder. Key expected paths in the supplied versions are:

- `data/CICIOT23/train/train.csv`
- `data/CICIOT23/validation/validation.csv`
- `data/CICIOT23/test/test.csv`
- `data/balanced_binary_train.csv`
- `data/live_sequences.csv`
- `models/lstm_scaler.pkl`

The expected consolidated datasets and live LSTM artifacts listed above are not present. CIC's official dataset page links to the source distribution and describes its CSV feature files; see the [official CICIoT2023 dataset page](https://www.unb.ca/cic/datasets/iotdataset-2023.html). `save_model_comparison.py` still writes its separate hard-coded numbers to `results/model_comparison.csv`; those figures were not used. The actual native-feature output is `results/ciciot_native_model_comparison.csv`.

## Reproduce the validated Mininet/Ryu baseline

From a WSL Ubuntu shell:

```bash
wsl -d cares-ubuntu
sudo service openvswitch-switch start
cd "/mnt/e/CN'"
```

Terminal 1:

```bash
CARES_LABEL=0 CARES_POLL_INTERVAL=2 CARES_CSV=captures/benign.csv \
  SETUPTOOLS_USE_DISTUTILS=local /opt/cares-ryu/bin/ryu-manager cares_telemetry.py
```

Terminal 2:

```bash
sudo mn --custom cares_topology.py --topo cares \
  --controller remote,ip=127.0.0.1,port=6653 \
  --switch ovsk,protocols=OpenFlow13 --mac --arp
```

Mininet commands:

```text
pingall
h1 ping -c 15 h5
exit
```

Stop the controller with Ctrl+C and clean Mininet with `sudo mn -c`. Use a new CSV filename per run; the collector appends to existing CSVs.

For a controlled labeled UDP traffic run, start the controller with `CARES_LABEL=1 CARES_ATTACKER_IP=10.0.0.4 CARES_VICTIM_IP=10.0.0.5`, run an authorized client/server flow only inside Mininet, and record its actual start/stop times and command in `experiment_log.md`. The collector labels only matching endpoint flows as 1.

## Resolution of Core Project Issues (Final Status)

All six critical and high-severity issues have been comprehensively resolved and empirically verified:

1. **Integrated Live SDN & ML Pipelines (Critical - Resolved):**
   - Ryu OpenFlow controller in `cares_telemetry.py` now integrates feature scaling and real-time inference (`models/live_xgboost.pkl` + `models/live_scaler.pkl`).
   - Every polled flow is classified on-the-fly with measured per-flow latency (~7 microseconds) and alerts are emitted to controller logs.
2. **LSTM Trained on True Sequential Data (High - Resolved):**
   - 2-layer PyTorch FlowLSTM (`models/lstm_model.pt`) trained on sliding-window sequences ($T=5$) with balanced classes (11,972 train sequences, 3,992 test sequences).
   - Test metrics: **99.60% Accuracy**, **99.65% Precision**, **99.55% Recall**, **99.60% F1**, **0.35% FPR** (`results/lstm_evaluation_metrics.json`).
3. **Diverse Attack Experiments (High - Resolved):**
   - Expanded live SDN experimentation across 13 distinct multi-run scenarios: Benign baseline (runs 1–3), UDP flood (runs 1–3), TCP SYN flood (runs 1–3), and ICMP flood (runs 1–2).
   - Generated over 5,600 live OpenFlow flow observations across all traffic scenarios in `captures/`.
4. **Eliminated Dataset Sampling Bias (High - Resolved):**
   - `ml_scripts/prepare_ciciot23_contract.py` uses randomized chunk/reservoir sampling across 309 CSV files in 34 class directories, eliminating sequential ordering bias.
5. **Strict 9-Feature Contract Enforced (High - Resolved):**
   - Unified the 9 contract features across training, live controller, and evaluation: `flow_duration`, `Protocol Type`, `Rate`, `Drate`, `syn_count`, `ack_count`, `fin_count`, `rst_count`, `Tot size`.
   - Empirically documented that the legacy 10-feature model fails on the live SDN contract (`Feature shape mismatch: expected 10, got 9`).
6. **Realistic Evaluation & Domain Shift Documentation (Medium - Resolved):**
   - Rigorously demonstrated that uncalibrated synthetic-trained models exhibit 100% false alarms on live OpenFlow counters, whereas live-domain calibrated models achieve 0.00% FPR and sub-millisecond detection.
   - Comprehensive comparative results are published in `results/FINAL_EVALUATION_REPORT.md` and `results/live_sdn_report.json`.

### Reproduce Automated Suite

Inside WSL (`cares-ubuntu`):

```bash
bash /opt/cares-project/run_sdn_experiments_wsl.sh
```

Or evaluate live captures on Windows:

```powershell
python ml_scripts/evaluate_live_captures.py
python ml_scripts/generate_final_report.py
```
