#!/bin/bash
# Helper to run LSTM training avoiding apostrophe path issues
cd /mnt/e/CN\'
source /opt/cares-venv/bin/activate
python3 ml_scripts/train_lstm_ciciot.py
