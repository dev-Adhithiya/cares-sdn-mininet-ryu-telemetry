
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
h1.cmd('ping -c 12 10.0.0.5 > /dev/null &')
h2.cmd('while true; do curl -s http://10.0.0.5:8080 > /dev/null; sleep 0.5; done &')
h3.cmd('iperf -s &')
h1.cmd('iperf -c 10.0.0.3 -t 12 -b 100K &')

# Generate attack if configured
attack_start_time = None
if False:
    time.sleep(3)  # 3s benign baseline before attack
    attack_start_time = time.time()
    print("ATTACK_START_TIME:", attack_start_time)
    if "none" == "udp":
        # UDP Flood attack from h4 to h5
        h4.cmd('iperf -c 10.0.0.5 -u -b 20M -t 9 &')
    elif "none" == "syn":
        # TCP SYN flood attack using hping3
        h4.cmd('hping3 -S -p 8080 --flood 10.0.0.5 2>/dev/null &')
    elif "none" == "icmp":
        # High-rate ICMP ping flood
        h4.cmd('ping -f -c 1000 10.0.0.5 > /dev/null &')
        
time.sleep(12)

net.stop()
