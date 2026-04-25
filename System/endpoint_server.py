import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import logging
import threading
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models for endpoint threat detection
endpoint_rf_model = joblib.load('EndPoint/Models/random_forest_model.pkl')
encoder = joblib.load('EndPoint/Models/label_encoder.pkl')

# Global variables to track packets and stats
flow_stats = defaultdict(lambda: {'src_bytes': 0, 'dst_bytes': 0})
total_packets = 0

# Function to simulate packet processing for demonstration
def simulate_packet_processing():
    while True:
        # Simulate processing packets every 5 seconds
        time.sleep(5)

        logging.info("Simulating packet processing...")

        # Simulated packet data
        packet_data = {
            'src_bytes': np.random.randint(0, 1000),
            'dst_bytes': np.random.randint(0, 1000)
        }

        flow_stats['src_bytes'] = packet_data['src_bytes']
        flow_stats['dst_bytes'] = packet_data['dst_bytes']

        # Process the simulated packet
        process_endpoint_packet(packet_data)

# Endpoint packet processing logic
def process_endpoint_packet(packet_data):
    global total_packets
    total_packets += 1

    logging.info("Processing packet...")  # Log to confirm packet processing

    # Use the packet data for prediction
    sample_data_input = [
        packet_data['src_bytes'], packet_data['dst_bytes'], 0, 0, 0, 0, 0, 0, 0, 0
    ]

    # Get prediction from the model
    rf_prediction = predict_from_models(sample_data_input)

    try:
        predicted_label = encoder.inverse_transform([rf_prediction])[0]
    except ValueError:
        predicted_label = rf_prediction

    # Emit the prediction to the client
    logging.info(f"Emitting prediction: {predicted_label}")  # Log emitted prediction
    socketio.emit('endpoint_threat', {'prediction': predicted_label})

# Prediction function
def predict_from_models(sample_data_input):
    sample_data = pd.DataFrame([sample_data_input], columns=[
        'src_bytes', 'dst_bytes', 'diff_srv_rate', 'dst_host_srv_count',
        'dst_host_srv_serror_rate', 'flag_SF', 'dst_host_diff_srv_rate',
        'dst_host_same_srv_rate', 'count', 'logged_in'
    ])

    # Scale the sample data
    X_train_relevant = pd.DataFrame([[0] * 10], columns=sample_data.columns)
    scaler_relevant = StandardScaler().fit(X_train_relevant)
    sample_data_scaled = scaler_relevant.transform(sample_data)

    # Get prediction from the random forest model
    rf_prediction = endpoint_rf_model.predict(sample_data_scaled)[0]
    return rf_prediction

# Flask route for rendering the endpoint threat detection page
@app.route('/endpoint', methods=['GET'])
def endpoint_threat():
    return render_template('endpoint.html')

# Start a background thread to simulate packet processing
def start_simulation():
    thread = threading.Thread(target=simulate_packet_processing)
    thread.start()

if __name__ == '__main__':
    # Start the simulation in a separate thread
    start_simulation()

    # Run Flask application with SocketIO
    socketio.run(app, port=5002, debug=True, allow_unsafe_werkzeug=True)
