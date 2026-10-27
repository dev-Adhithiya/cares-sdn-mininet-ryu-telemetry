# Experiment log

## Benign baseline capture

- Run/capture name: `benign`
- Scenario: benign only
- Attack type: none
- Attacker host/IP: none
- Victim/server host/IP: h5 / 10.0.0.5
- Observation window (UTC): 2026-10-11 16:31:03 to 16:31:17
- Commands: `pingall`; `h1 ping -c 15 h5`
- Background traffic: Mininet ARP/ICMP from pingall
- Poll interval: 1 second
- CSV: `captures/benign.csv`
- Result: pingall 0% loss (20/20); 312 observations across 21 flow IDs, each with at least 14 polls, stable flow IDs recur across polls, labels are 0.

## Controlled UDP attack capture

- Run/capture name: `udp_attack`
- Scenario: controlled UDP traffic from h4 to h5
- Attack type: UDP load, 64 Kbit/s for 15 seconds (isolated Mininet lab)
- Attacker host/IP: h4 / 10.0.0.4
- Victim/server host/IP: h5 / 10.0.0.5
- Attack start time (UTC): 2026-10-11 16:32:05
- Attack stop time (UTC): 2026-10-11 16:32:21
- Commands: `h5 iperf -s -u -p 5001 -D`; `h4 iperf -c 10.0.0.5 -u -p 5001 -b 64K -t 15`
- Background traffic: none beyond setup/ARP
- Poll interval: 1 second
- Random seed/configuration: none
- CSV: `captures/udp_attack.csv`
- Result: iperf sent 88 datagrams, 0% packet loss; attacker/victim UDP flow has 15 observations labelled 1. Unmatched/table-miss observations have blank labels.

## Environment

- WSL 2 distro: Ubuntu 24.04.5 LTS (`cares-ubuntu`)
- Mininet: 2.3.0
- Open vSwitch: 3.3.9
- Ryu: 4.34, installed in `/opt/cares-ryu` with the compatibility steps in `setup_wsl.sh`

