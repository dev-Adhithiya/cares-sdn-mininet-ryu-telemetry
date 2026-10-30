import sys
import os
import time
from functools import partial

sys.path.insert(0, "/usr/lib/python3/dist-packages")
sys.path.insert(0, "/opt/cares-venv/lib/python3.12/site-packages")
sys.path.insert(0, "/opt/cares-project")

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.log import setLogLevel
from cares_topology import CaresTopo

setLogLevel("info")

topo = CaresTopo()
SW13 = partial(OVSSwitch, protocols="OpenFlow13")
RC = partial(RemoteController, ip="127.0.0.1", port=6653)
net = Mininet(topo=topo, controller=RC, switch=SW13, autoSetMacs=True, autoStaticArp=True)

print("Starting Mininet...")
net.start()
print("Pinging all...")
loss = net.pingAll()
print(f"PingAll loss: {loss}%")
time.sleep(3)
net.stop()
print("Done.")
