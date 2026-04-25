import logging
import scapy.all as scapy
import threading
import numpy as np
import joblib
from collections import defaultdict, deque
from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load the saved model and encoder
model = joblib.load('Network/Models/gan_Predict.pkl')
encoder = joblib.load('Network/Models/label_encoder.pkl')

# Global variables to track packets and flow statistics
flow_stats = defaultdict(lambda: {
    'total_fwd_packets': 0,
    'total_bwd_packets': 0,
    'total_length_fwd': 0,
    'total_length_bwd': 0,
    'fwd_packet_lengths': [],
    'bwd_packet_lengths': [],
    'flow_start_time': None,
    'flow_end_time': None,
    'flow_iat': [],
    'fwd_iat': [],
    'bwd_iat': [],
    'fwd_flags': {'PSH': 0, 'URG': 0, 'FIN': 0, 'SYN': 0, 'RST': 0, 'ACK': 0, 'CWE': 0, 'ECE': 0},
    'bwd_flags': {'PSH': 0, 'URG': 0, 'FIN': 0, 'SYN': 0, 'RST': 0, 'ACK': 0, 'CWE': 0, 'ECE': 0},
    'fwd_header_length': 0,
    'bwd_header_length': 0,
    'init_win_bytes_fwd': 0,
    'init_win_bytes_bwd': 0,
    'act_data_pkt_fwd': 0,
    'min_seg_size_fwd': float('inf'),
    'idle_times': [],
    'fwd_bulk_data': [],
    'bwd_bulk_data': [],
    'flow_duration': 0,
    'protocol': None,
})

total_packets = 0
non_ip_packet_count = 0
class_counts = defaultdict(int)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cleanup function to remove old or completed flows
def cleanup_flow_stats():
    current_time = time.time()
    for flow_key in list(flow_stats.keys()):
        flow_duration = current_time - flow_stats[flow_key]['flow_end_time']
        if flow_duration > 600:
            del flow_stats[flow_key]

# Packet feature extraction and statistics calculation
def calculate_statistics(packet):
    pkt_time = time.time()
    src_ip = packet[scapy.IP].src
    dst_ip = packet[scapy.IP].dst
    proto = packet[scapy.IP].proto
    length = len(packet)
    direction = 'fwd' if packet[scapy.IP].src == src_ip else 'bwd'
    flow_key = (src_ip, dst_ip, proto)

    flow_stats[flow_key]['protocol'] = proto

    if flow_stats[flow_key]['flow_start_time'] is None:
        flow_stats[flow_key]['flow_start_time'] = pkt_time

    flow_stats[flow_key]['flow_end_time'] = pkt_time
    flow_stats[flow_key]['flow_duration'] = flow_stats[flow_key]['flow_end_time'] - flow_stats[flow_key][
        'flow_start_time']
    flow_stats[flow_key][f'total_{direction}_packets'] += 1
    flow_stats[flow_key][f'total_length_{direction}'] += length
    flow_stats[flow_key][f'{direction}_packet_lengths'].append(length)
    flow_stats[flow_key][f'{direction}_iat'].append(pkt_time)

    if scapy.TCP in packet:
        tcp_flags = packet[scapy.TCP].flags
        flow_stats[flow_key][f'{direction}_flags']['FIN'] += 1 if tcp_flags.F else 0
        flow_stats[flow_key][f'{direction}_flags']['SYN'] += 1 if tcp_flags.S else 0
        flow_stats[flow_key][f'{direction}_flags']['RST'] += 1 if tcp_flags.R else 0
        flow_stats[flow_key][f'{direction}_flags']['PSH'] += 1 if tcp_flags.P else 0
        flow_stats[flow_key][f'{direction}_flags']['ACK'] += 1 if tcp_flags.A else 0
        flow_stats[flow_key][f'{direction}_flags']['URG'] += 1 if tcp_flags.U else 0
        flow_stats[flow_key][f'{direction}_flags']['CWE'] += 1 if tcp_flags.C else 0
        flow_stats[flow_key][f'{direction}_flags']['ECE'] += 1 if tcp_flags.E else 0
        flow_stats[flow_key][f'{direction}_header_length'] += packet[scapy.TCP].dataofs * 4

# Extract selected features for model prediction
def extract_selected_features(flow_key, stats):
    fwd_packet_lengths = stats['fwd_packet_lengths']
    bwd_packet_lengths = stats['bwd_packet_lengths']
    fwd_iat = np.diff(stats['fwd_iat'])
    bwd_iat = np.diff(stats['bwd_iat'])

    selected_features = {
        'Fwd Packet Length Std': np.std(fwd_packet_lengths) if fwd_packet_lengths else 0,
        'Bwd Packet Length Std': np.std(bwd_packet_lengths) if bwd_packet_lengths else 0,
        'Fwd IAT Mean': np.mean(fwd_iat) if fwd_iat.size > 0 else 0,
        'Fwd IAT Std': np.std(fwd_iat) if fwd_iat.size > 0 else 0,
        'Fwd IAT Max': np.max(fwd_iat) if fwd_iat.size > 0 else 0,
        'Bwd IAT Std': np.std(bwd_iat) if bwd_iat.size > 0 else 0,
        'Bwd Packets/s': stats['total_bwd_packets'] / stats['flow_duration'] if stats['flow_duration'] > 0 else 0,
        'Idle Mean': np.mean(stats['idle_times']) if stats['idle_times'] else 0,
        'Idle Max': np.max(stats['idle_times']) if stats['idle_times'] else 0,
        'Idle Min': np.min(stats['idle_times']) if stats['idle_times'] else 0,
    }

    return selected_features

# Packet processing and prediction
def process_packet(packet, interface):
    global total_packets, non_ip_packet_count, class_counts

    try:
        if packet.haslayer(scapy.IP):
            total_packets += 1
            calculate_statistics(packet)

            src_ip = packet[scapy.IP].src
            dst_ip = packet[scapy.IP].dst
            proto = packet[scapy.IP].proto
            flow_key = (src_ip, dst_ip, proto)
            stats = flow_stats[flow_key]

            selected_features = extract_selected_features(flow_key, stats)
            feature_values = np.array(list(selected_features.values())).reshape(1, -1)

            # Predict using the model
            prediction = model.predict(feature_values)
            predicted_label = encoder.inverse_transform(prediction)

            logging.info(f"Interface: {interface} - Predicted Class: {predicted_label[0]}")

            class_counts[predicted_label[0]] += 1

            socketio.emit('network_threat', {
                'prediction': predicted_label[0],
                'total_packets': total_packets,
                'non_ip_packet_count': non_ip_packet_count,
                'class_counts': dict(class_counts)
            })

            cleanup_flow_stats()

        else:
            non_ip_packet_count += 1
            class_counts["Non-IP Packet"] += 1
            logging.info(f"Non-IP packet processed.")
            socketio.emit('network_threat', {
                'prediction': 'Non-IP packet',
                'total_packets': total_packets,
                'non_ip_packet_count': non_ip_packet_count,
                'class_counts': dict(class_counts)
            })

    except Exception as e:
        logging.error(f"Error processing packet: {str(e)}")

# Sniff packets on the interface
def sniff_interface(interface):
    try:
        logging.info(f"Starting sniffing on interface: {interface}")
        scapy.sniff(iface=interface, prn=lambda pkt: process_packet(pkt, interface), store=False)
    except Exception as e:
        logging.error(f"Error sniffing on interface {interface}: {str(e)}")

# Start sniffing on all interfaces
def start_sniffing():
    try:
        interfaces = scapy.get_if_list()
        logging.info(f"Available interfaces: {interfaces}")

        threads = []
        for interface in interfaces:
            thread = threading.Thread(target=sniff_interface, args=(interface,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    except Exception as e:
        logging.critical(f"Error in sniffing: {str(e)}")

# Flask route
@app.route('/network', methods=['GET'])
def network_threat():
    threading.Thread(target=start_sniffing).start()
    return render_template('network.html')

# Start the Flask app
if __name__ == '__main__':
    socketio.run(app, port=5005, debug=True, allow_unsafe_werkzeug=True)

