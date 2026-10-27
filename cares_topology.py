from mininet.topo import Topo


class CaresTopo(Topo):
    """One-switch CARES-SDN baseline: h1-h3 benign, h4 attacker, h5 victim."""

    def build(self):
        switch = self.addSwitch("s1", protocols="OpenFlow13")
        hosts = [self.addHost(f"h{i}") for i in range(1, 6)]
        for host in hosts:
            self.addLink(host, switch)


topos = {"cares": CaresTopo}
