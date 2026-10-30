"""OpenFlow 1.3 learning switch and periodic telemetry stats collector with live ML detector.

The controller uses OpenStack OS-Ken / Ryu API.
Run with: osken-manager cares_telemetry.py or ryu-manager cares_telemetry.py
"""

import csv
import os
import sys
import time
from datetime import datetime, timezone

# Ensure /opt/cares-venv packages can be imported if running inside WSL
for extra_path in ["/opt/cares-venv/lib/python3.12/site-packages", "E:/CN'/models"]:
    if os.path.exists(extra_path) and extra_path not in sys.path:
        sys.path.insert(0, extra_path)

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ipv4, ipv6, packet, tcp, udp
from ryu.ofproto import ofproto_v1_3

CSV_FIELDS = [
    "flow_id", "timestamp", "duration_sec", "duration_nsec", "packet_count",
    "byte_count", "protocol", "src_ip", "dst_ip", "src_port", "dst_port",
    "poll_interval", "flow_duration", "Protocol Type", "Rate", "Drate",
    "syn_count", "ack_count", "fin_count", "rst_count", "Tot size",
    "binary_label", "predicted_label", "prediction_confidence", "inference_latency_ms"
]

SEQUENCE_FIELDS = [
    "flow_id", "timestamp", "flow_duration", "Protocol Type", "Rate", "Drate",
    "syn_count", "ack_count", "fin_count", "rst_count", "Tot size", "binary_label"
]


def _match_value(match, key):
    value = match.get(key)
    return "" if value is None else value


class CaresTelemetry(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(CaresTelemetry, self).__init__(*args, **kwargs)
        self.poll_interval = float(os.getenv("CARES_POLL_INTERVAL", "1.0"))
        if self.poll_interval <= 0:
            raise ValueError("CARES_POLL_INTERVAL must be positive")
        self.csv_path = os.getenv("CARES_CSV", "raw_telemetry.csv")
        self.sequences_path = os.getenv("CARES_LIVE_SEQUENCES", "data/live_sequences.csv")
        
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
        self.flow_history = {}  # flow_id -> dict(time, packet_count, byte_count, duration)
        self.flow_flags = {}    # flow_id -> dict(syn, ack, fin, rst)
        
        # Load ML Detector
        self.model = None
        self.scaler = None
        self.model_path = os.getenv("CARES_MODEL", "models/live_xgboost.pkl")
        self.scaler_path = os.getenv("CARES_SCALER", "")
        self._load_detector()
        
        self.monitor_thread = hub.spawn(self._poll)
        self._ensure_csv()

    def _load_detector(self):
        """Load trained ML model and matching scaler if available."""
        import sys
        for p in ["/opt/cares-venv/lib/python3.12/site-packages", "/mnt/e/CN'", "E:/CN'"]:
            if os.path.exists(p) and p not in sys.path:
                sys.path.insert(0, p)
                
        candidates = [
            self.model_path,
            "models/live_xgboost.pkl",
            "models/live_random_forest.pkl",
            "models/random_forest_model.pkl",
            "models/xgboost_model.pkl"
        ]
        loaded_model_path = None
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    import joblib
                    self.model = joblib.load(cand)
                    self.logger.info("Successfully loaded ML Detector from %s", cand)
                    loaded_model_path = cand
                    break
                except Exception as e:
                    self.logger.warning("Failed to load model from %s: %s", cand, e)

        if loaded_model_path is None:
            self.logger.info("No pre-trained model found. Live detector in data-collection mode.")
            return

        # Load appropriate scaler
        scaler_candidates = [self.scaler_path] if self.scaler_path else []
        if "live" in loaded_model_path:
            scaler_candidates.extend(["models/live_scaler.pkl", "models/lstm_scaler.pkl"])
        else:
            scaler_candidates.extend(["models/lstm_scaler.pkl", "models/live_scaler.pkl"])

        for sc in scaler_candidates:
            if sc and os.path.exists(sc):
                try:
                    import joblib
                    self.scaler = joblib.load(sc)
                    self.logger.info("Successfully loaded Feature Scaler from %s", sc)
                    break
                except Exception as e:
                    self.logger.warning("Failed to load scaler from %s: %s", sc, e)

    def _ensure_csv(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=CSV_FIELDS).writeheader()
                
        if self.sequences_path:
            os.makedirs(os.path.dirname(self.sequences_path) or ".", exist_ok=True)
            if not os.path.exists(self.sequences_path) or os.path.getsize(self.sequences_path) == 0:
                with open(self.sequences_path, "w", newline="", encoding="utf-8") as output:
                    csv.DictWriter(output, fieldnames=SEQUENCE_FIELDS).writeheader()

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

        ip4 = pkt.get_protocol(ipv4.ipv4)
        ip6 = pkt.get_protocol(ipv6.ipv6)
        match_fields = {"in_port": in_port, "eth_src": eth.src, "eth_dst": eth.dst}
        
        tcp_seg = pkt.get_protocol(tcp.tcp)
        udp_seg = pkt.get_protocol(udp.udp)
        
        # Track TCP flags for flow
        flow_key = None
        if ip4:
            match_fields.update(eth_type=0x0800, ipv4_src=ip4.src,
                                ipv4_dst=ip4.dst, ip_proto=ip4.proto)
            if tcp_seg:
                match_fields.update(tcp_src=tcp_seg.src_port, tcp_dst=tcp_seg.dst_port)
                flow_key = (ip4.src, ip4.dst, 6, tcp_seg.src_port, tcp_seg.dst_port)
            elif udp_seg:
                match_fields.update(udp_src=udp_seg.src_port, udp_dst=udp_seg.dst_port)
                flow_key = (ip4.src, ip4.dst, 17, udp_seg.src_port, udp_seg.dst_port)
            else:
                flow_key = (ip4.src, ip4.dst, ip4.proto, 0, 0)
        elif ip6:
            match_fields.update(eth_type=0x86DD, ipv6_src=ip6.src,
                                ipv6_dst=ip6.dst, ip_proto=ip6.nxt)
            if tcp_seg:
                match_fields.update(tcp_src=tcp_seg.src_port, tcp_dst=tcp_seg.dst_port)
            elif udp_seg:
                match_fields.update(udp_src=udp_seg.src_port, udp_dst=udp_seg.dst_port)
        else:
            match_fields.update(eth_type=eth.ethertype)

        # Inspect flags if TCP
        if tcp_seg and flow_key:
            flags = self.flow_flags.setdefault(flow_key, {"syn": 0, "ack": 0, "fin": 0, "rst": 0})
            bits = tcp_seg.bits
            if bits & 0x02:
                flags["syn"] += 1
            if bits & 0x10:
                flags["ack"] += 1
            if bits & 0x01:
                flags["fin"] += 1
            if bits & 0x04:
                flags["rst"] += 1

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
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat(timespec="milliseconds")
        rows = []
        seq_rows = []
        
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
            src_port = _match_value(match, "tcp_src") or _match_value(match, "udp_src")
            dst_port = _match_value(match, "tcp_dst") or _match_value(match, "udp_dst")
            
            # Ground truth labeling
            if self.binary_label == "1":
                is_attack_flow = {src_ip, dst_ip} == {self.attacker_ip, self.victim_ip}
                row_label = "1" if is_attack_flow else ""
            else:
                row_label = self.binary_label
                
            # -------------------------------------------------------------
            # Feature Extraction (9 Contract Features)
            # -------------------------------------------------------------
            cur_duration = float(stat.duration_sec) + float(stat.duration_nsec) * 1e-9
            cur_packets = int(stat.packet_count)
            cur_bytes = int(stat.byte_count)
            
            prev_stat = self.flow_history.get(flow_id)
            if prev_stat is not None:
                dt = max(cur_duration - prev_stat["duration"], 0.001)
                d_packets = max(cur_packets - prev_stat["packet_count"], 0)
                d_bytes = max(cur_bytes - prev_stat["byte_count"], 0)
            else:
                dt = max(cur_duration, 0.001)
                d_packets = cur_packets
                d_bytes = cur_bytes
                
            rate = d_packets / dt
            drate = d_bytes / dt
            tot_size = d_bytes / max(d_packets, 1)
            proto_val = int(protocol) if str(protocol).isdigit() else 0
            
            # Lookup TCP flags
            lookup_key = (src_ip, dst_ip, proto_val, int(src_port or 0), int(dst_port or 0))
            flags = self.flow_flags.get(lookup_key, {"syn": 0, "ack": 0, "fin": 0, "rst": 0})
            
            # Update history
            self.flow_history[flow_id] = {
                "duration": cur_duration,
                "packet_count": cur_packets,
                "byte_count": cur_bytes,
                "time": time.time()
            }
            
            # -------------------------------------------------------------
            # Real-Time ML Model Inference
            # -------------------------------------------------------------
            pred_label = ""
            confidence = ""
            latency_ms = ""
            
            if self.model is not None and (src_ip and dst_ip):
                feature_vector = [
                    cur_duration,
                    proto_val,
                    rate,
                    drate,
                    flags["syn"],
                    flags["ack"],
                    flags["fin"],
                    flags["rst"],
                    tot_size
                ]
                
                t_start = time.perf_counter()
                try:
                    import numpy as np
                    inp = np.array([feature_vector], dtype=np.float32)
                    if self.scaler is not None:
                        inp = self.scaler.transform(inp)
                    pred = int(self.model.predict(inp)[0])
                    t_latency = (time.perf_counter() - t_start) * 1000.0
                    pred_label = str(pred)
                    latency_ms = f"{t_latency:.3f}"
                    
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(inp)[0]
                        conf_val = float(proba[pred])
                        confidence = f"{conf_val:.4f}"
                    else:
                        confidence = "1.0000"
                        
                    if pred == 1:
                        self.logger.warning(
                            "[ALERT - INTRUSION DETECTED] Flow=%s %s -> %s Proto=%s Rate=%.1f pps Conf=%s Latency=%sms",
                            flow_id, src_ip, dst_ip, protocol, rate, confidence, latency_ms
                        )
                except Exception as err:
                    self.logger.error("Inference error on flow %s: %s", flow_id, err)
                    
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
                "src_port": src_port,
                "dst_port": dst_port,
                "poll_interval": self.poll_interval,
                "flow_duration": f"{cur_duration:.4f}",
                "Protocol Type": proto_val,
                "Rate": f"{rate:.2f}",
                "Drate": f"{drate:.2f}",
                "syn_count": flags["syn"],
                "ack_count": flags["ack"],
                "fin_count": flags["fin"],
                "rst_count": flags["rst"],
                "Tot size": f"{tot_size:.2f}",
                "binary_label": row_label,
                "predicted_label": pred_label,
                "prediction_confidence": confidence,
                "inference_latency_ms": latency_ms,
            })
            
            # If flow has valid endpoints, also add to sequence data
            if src_ip and dst_ip and row_label != "":
                seq_rows.append({
                    "flow_id": flow_id,
                    "timestamp": now,
                    "flow_duration": f"{cur_duration:.4f}",
                    "Protocol Type": proto_val,
                    "Rate": f"{rate:.2f}",
                    "Drate": f"{drate:.2f}",
                    "syn_count": flags["syn"],
                    "ack_count": flags["ack"],
                    "fin_count": flags["fin"],
                    "rst_count": flags["rst"],
                    "Tot size": f"{tot_size:.2f}",
                    "binary_label": row_label,
                })
                
        if rows:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as output:
                csv.DictWriter(output, fieldnames=CSV_FIELDS).writerows(rows)
            self.logger.info("Wrote %d flow observations to %s", len(rows), self.csv_path)
            
        if seq_rows and self.sequences_path:
            with open(self.sequences_path, "a", newline="", encoding="utf-8") as seq_out:
                csv.DictWriter(seq_out, fieldnames=SEQUENCE_FIELDS).writerows(seq_rows)
