"""OpenFlow 1.3 learning switch and periodic raw stats collector.

The controller uses OpenStack OS-Ken, a maintained Ryu API-compatible fork.
Run with: osken-manager cares_telemetry.py
"""

import csv
import os
from datetime import datetime, timezone

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ipv4, ipv6, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3

CSV_FIELDS = [
    "flow_id", "timestamp", "duration_sec", "duration_nsec", "packet_count",
    "byte_count", "protocol", "src_ip", "dst_ip", "src_port", "dst_port",
    "poll_interval", "binary_label",
]


def _match_value(match, key):
    value = match.get(key)
    return "" if value is None else value


class CaresTelemetry(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CaresTelemetry, self).__init__(*args, **kwargs)
        self.poll_interval = float(os.getenv("CARES_POLL_INTERVAL", "2"))
        if self.poll_interval <= 0:
            raise ValueError("CARES_POLL_INTERVAL must be positive")
        self.csv_path = os.getenv("CARES_CSV", "raw_telemetry.csv")
        label = os.getenv("CARES_LABEL", "").strip()
        if label not in ("", "0", "1"):
            raise ValueError("CARES_LABEL must be unset, 0, or 1")
        self.binary_label = label
        self.attacker_ip = os.getenv("CARES_ATTACKER_IP", "").strip()
        self.victim_ip = os.getenv("CARES_VICTIM_IP", "").strip()
        if label == "1" and not (self.attacker_ip and self.victim_ip):
            raise ValueError("CARES_ATTACKER_IP and CARES_VICTIM_IP are required when CARES_LABEL=1")
        self.datapaths = {}
        self.flow_ids = {}
        self.mac_to_port = {}
        self.monitor_thread = hub.spawn(self._poll)
        self._ensure_csv()

    def _ensure_csv(self):
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=CSV_FIELDS).writeheader()

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                                      actions)]
        datapath.send_msg(parser.OFPFlowMod(datapath=datapath, priority=0,
                                            match=match, instructions=instructions))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[datapath.id] = datapath
            self.logger.info("OpenFlow switch connected: dpid=%016x", datapath.id)
        elif datapath.id in self.datapaths:
            del self.datapaths[datapath.id]

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == 0x88cc:
            return

        dpid = datapath.id
        learned = self.mac_to_port.setdefault(dpid, {})
        learned[eth.src] = in_port
        out_port = learned.get(eth.dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        # Install directional IP 5-tuple rules when transport ports are present;
        # retain IP-only matches for ICMP and other protocols.
        ip4 = pkt.get_protocol(ipv4.ipv4)
        ip6 = pkt.get_protocol(ipv6.ipv6)
        match_fields = {"in_port": in_port, "eth_src": eth.src, "eth_dst": eth.dst}
        if ip4:
            match_fields.update(eth_type=0x0800, ipv4_src=ip4.src,
                                ipv4_dst=ip4.dst, ip_proto=ip4.proto)
            tcp_seg = pkt.get_protocol(tcp.tcp)
            udp_seg = pkt.get_protocol(udp.udp)
            if tcp_seg:
                match_fields.update(tcp_src=tcp_seg.src_port, tcp_dst=tcp_seg.dst_port)
            elif udp_seg:
                match_fields.update(udp_src=udp_seg.src_port, udp_dst=udp_seg.dst_port)
        elif ip6:
            match_fields.update(eth_type=0x86DD, ipv6_src=ip6.src,
                                ipv6_dst=ip6.dst, ip_proto=ip6.nxt)
            tcp_seg = pkt.get_protocol(tcp.tcp)
            udp_seg = pkt.get_protocol(udp.udp)
            if tcp_seg:
                match_fields.update(tcp_src=tcp_seg.src_port, tcp_dst=tcp_seg.dst_port)
            elif udp_seg:
                match_fields.update(udp_src=udp_seg.src_port, udp_dst=udp_seg.dst_port)
        else:
            match_fields.update(eth_type=eth.ethertype)

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(**match_fields)
            instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                                          actions)]
            datapath.send_msg(parser.OFPFlowMod(
                datapath=datapath, priority=10, idle_timeout=30,
                match=match, instructions=instructions))

        data = None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id, in_port=in_port,
            actions=actions, data=data))

    def _poll(self):
        while True:
            for datapath in list(self.datapaths.values()):
                parser = datapath.ofproto_parser
                datapath.send_msg(parser.OFPFlowStatsRequest(datapath))
            hub.sleep(self.poll_interval)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        datapath = ev.msg.datapath
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        rows = []
        for stat in ev.msg.body:
            match = stat.match
            identity = (datapath.id, stat.table_id, stat.priority,
                        tuple(sorted((str(k), str(v)) for k, v in match.items())))
            flow_id = self.flow_ids.setdefault(
                identity, "flow-%016x-%06d" % (datapath.id, len(self.flow_ids) + 1))
            eth_type = match.get("eth_type")
            protocol = match.get("ip_proto") if eth_type in (0x0800, 0x86DD) else ""
            src_ip = _match_value(match, "ipv4_src") or _match_value(match, "ipv6_src")
            dst_ip = _match_value(match, "ipv4_dst") or _match_value(match, "ipv6_dst")
            if self.binary_label == "1":
                is_attack_flow = {src_ip, dst_ip} == {self.attacker_ip, self.victim_ip}
                row_label = "1" if is_attack_flow else ""
            else:
                row_label = self.binary_label
            rows.append({
                "flow_id": flow_id,
                "timestamp": now,
                "duration_sec": stat.duration_sec,
                "duration_nsec": stat.duration_nsec,
                "packet_count": stat.packet_count,
                "byte_count": stat.byte_count,
                "protocol": protocol if protocol is not None else "",
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": _match_value(match, "tcp_src") or _match_value(match, "udp_src"),
                "dst_port": _match_value(match, "tcp_dst") or _match_value(match, "udp_dst"),
                "poll_interval": self.poll_interval,
                "binary_label": row_label,
            })
        if rows:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=CSV_FIELDS).writerows(rows)
            self.logger.info("Wrote %d flow observations to %s", len(rows), self.csv_path)

