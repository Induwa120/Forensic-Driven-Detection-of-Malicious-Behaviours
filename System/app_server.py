import logging
import cv2
import joblib
import keras
import numpy as np
import pandas as pd
import scapy.all as scapy
import tensorflow as tf
import threading
import time
from flask import Flask, render_template, request, Response, jsonify
from flask_socketio import SocketIO
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Load models
behavioral_model = tf.keras.models.load_model('Behavioral/Models/cnn_cert-human-b_model.h5')
behavioral_scaler = joblib.load('Behavioral/Models/scaler.pkl')
endpoint_rf_model = joblib.load('EndPoint/Models/random_forest_model.pkl')
network_model = joblib.load('Network/Models/random_forest_classifier.pkl')
network_encoder = joblib.load('Network/Models/label_encoder.pkl')
physical_model = keras.models.load_model('Physical/Models/physical_s_model.keras')

# Global variables for packet tracking and flow statistics
flow_stats = defaultdict(lambda: {
    'total_fwd_packets': 0,
    'total_bwd_packets': 0,
    'total_length_fwd': 0,
    'total_length_bwd': 0,
    'flow_start_time': None,
    'flow_end_time': None,
    'protocol': None,
})
total_packets = 0
class_counts = defaultdict(int)
non_ip_packet_count = 0

# Lock for thread safety
lock = threading.Lock()

# Logging setup
logging.basicConfig(level=logging.INFO)


# Sniffing logic
def process_packet(packet):
    global total_packets, non_ip_packet_count
    with lock:
        total_packets += 1
    if packet.haslayer(scapy.IP):
        calculate_flow_statistics(packet)
        process_endpoint_packet(packet)
        with lock:
            socketio.emit('network_threat', {
                'prediction': "Network Packet",
                'total_packets': total_packets,
                'class_counts': dict(class_counts),
                'non_ip_packet_count': non_ip_packet_count
            })
    else:
        with lock:
            non_ip_packet_count += 1
            socketio.emit('network_threat', {
                'prediction': 'Non-IP packet',
                'total_packets': total_packets,
                'class_counts': dict(class_counts),
                'non_ip_packet_count': non_ip_packet_count
            })


def calculate_flow_statistics(packet):
    src_ip = packet[scapy.IP].src
    dst_ip = packet[scapy.IP].dst
    proto = packet[scapy.IP].proto
    length = len(packet)
    flow_key = (src_ip, dst_ip, proto)
    with lock:
        flow_stats[flow_key]['protocol'] = proto
        flow_stats[flow_key]['flow_start_time'] = flow_stats[flow_key].get('flow_start_time', time.time())
        flow_stats[flow_key]['flow_end_time'] = time.time()
        direction = 'fwd' if packet[scapy.IP].src == src_ip else 'bwd'
        flow_stats[flow_key][f'total_{direction}_packets'] += 1
        flow_stats[flow_key][f'total_length_{direction}'] += length


def process_endpoint_packet(packet):
    """Specific processing logic for endpoint threat detection."""
    if packet.haslayer(scapy.IP):
        flow_key = calculate_endpoint_features(packet)
        stats = flow_stats[flow_key]
        sample_data_input = [
            stats['total_length_fwd'], stats['total_length_bwd'], 0, 0, 0, 0, 0, 0, 0, 0  # Example data structure
        ]
        # Call prediction function
        rf_prediction = predict_from_models(sample_data_input)
        try:
            predicted_label = network_encoder.inverse_transform([rf_prediction])[0]
        except ValueError:
            predicted_label = rf_prediction
        with lock:
            socketio.emit('endpoint_threat', {
                'prediction': predicted_label,
                'src_ip': packet[scapy.IP].src,
                'dst_ip': packet[scapy.IP].dst
            })


def calculate_endpoint_features(packet):
    """Extract features for endpoint threat detection."""
    src_ip = packet[scapy.IP].src
    dst_ip = packet[scapy.IP].dst
    proto = packet[scapy.IP].proto
    length = len(packet)
    flow_key = (src_ip, dst_ip, proto)
    with lock:
        flow_stats[flow_key]['flow_start_time'] = flow_stats[flow_key].get('flow_start_time', time.time())
        flow_stats[flow_key]['flow_end_time'] = time.time()
        direction = 'fwd' if packet[scapy.IP].src == src_ip else 'bwd'
        flow_stats[flow_key][f'total_{direction}_packets'] += 1
        flow_stats[flow_key][f'total_length_{direction}'] += length
    return flow_key


def start_sniffing():
    """Unified packet sniffing for both network and endpoint."""
    try:
        interfaces = scapy.get_if_list()
        logging.info(f"Available interfaces: {interfaces}")
        for interface in interfaces:
            logging.info(f"Sniffing on interface: {interface}")
            scapy.sniff(iface=interface, prn=process_packet, store=False)
    except Exception as e:
        logging.error(f"Error in sniffing: {str(e)}")


@app.route('/', methods=['GET'])
def index():
    threading.Thread(target=start_sniffing).start()
    return render_template('index.html')

def predict_from_models(sample_data_input):
    sample_data = pd.DataFrame([sample_data_input], columns=[
        'src_bytes', 'dst_bytes', 'diff_srv_rate', 'dst_host_srv_count',
        'dst_host_srv_serror_rate', 'flag_SF', 'dst_host_diff_srv_rate',
        'dst_host_same_srv_rate', 'count', 'logged_in'
    ])
    X_train_relevant = pd.DataFrame([[0] * 10], columns=sample_data.columns)
    scaler_relevant = StandardScaler().fit(X_train_relevant)
    sample_data_scaled = scaler_relevant.transform(sample_data)
    rf_prediction = endpoint_rf_model.predict(sample_data_scaled)[0]
    return rf_prediction


@socketio.on('connect')
def handle_connect():
    print("Client connected")


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
