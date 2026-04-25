import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RealTimeAnomalyDetector:
    def __init__(self):
        self.file_model = None
        self.mouse_model = None
        self.load_models()

    def load_models(self):
        """Load both file access and mouse movement models"""
        try:
            # Load file access model
            file_model_path = 'file/best_model.h5'
            self.file_model = load_model(file_model_path)
            logging.info("File access model loaded successfully")

            # Load mouse movement model
            mouse_model_path = 'file/lstm_autoencoder_model.h5'
            self.mouse_model = load_model(mouse_model_path,
                                          custom_objects={'mse': MeanSquaredError()})
            logging.info("Mouse movement model loaded successfully")

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")

    def process_file_access(self, file_data):
        """Process real-time file access data"""
        try:
            if self.file_model is None:
                logging.error("File access model not loaded")
                return False, 0.0

            features = file_data.get('features', [])
            if not features:
                return False, 0.0

            # Convert to numpy array
            features_array = np.array(features, dtype=np.float32)

            # Ensure correct shape
            if len(features_array.shape) == 1:
                features_array = features_array.reshape(1, -1)

            # Make prediction
            prediction = self.file_model.predict(features_array, verbose=0)
            anomaly_score = float(prediction[0][0] if len(prediction[0]) > 1 else prediction[0])

            # Determine anomaly (adjust threshold as needed)
            is_anomaly = anomaly_score > 0.5

            logging.info(f"File access - Anomaly: {is_anomaly}, Score: {anomaly_score:.4f}")
            return is_anomaly, anomaly_score

        except Exception as e:
            logging.error(f"Error processing file access: {str(e)}")
            return False, 0.0

    def process_mouse_movement(self, mouse_data):
        """Process real-time mouse movement data"""
        try:
            if self.mouse_model is None:
                logging.error("Mouse movement model not loaded")
                return False, 0.0

            movements = mouse_data.get('movements', [])
            if len(movements) < 10:
                return False, 0.0

            # Use last 10 movements for prediction
            input_data = np.array(movements[-10:], dtype=np.float32)

            # Normalize data
            mean_val = np.mean(input_data, axis=0, keepdims=True)
            std_val = np.std(input_data, axis=0, keepdims=True)
            std_val[std_val == 0] = 1

            input_data = (input_data - mean_val) / std_val

            # Reshape for LSTM
            input_data = input_data.reshape(1, 10, 3)

            # Get reconstruction
            reconstructed = self.mouse_model.predict(input_data, verbose=0)
            reconstruction_error = np.mean(np.abs(reconstructed - input_data))

            # Determine anomaly (adjust threshold as needed)
            is_anomaly = reconstruction_error > 1.5

            logging.info(f"Mouse movement - Anomaly: {is_anomaly}, Error: {reconstruction_error:.4f}")
            return is_anomaly, float(reconstruction_error)

        except Exception as e:
            logging.error(f"Error processing mouse movement: {str(e)}")
            return False, 0.0


# Global detector instance
detector = RealTimeAnomalyDetector()


# Legacy functions for backward compatibility
def predict_anomaly(mouse_data):
    """Legacy function for mouse anomaly prediction"""
    is_anomaly, error = detector.process_mouse_movement({'movements': mouse_data})
    return is_anomaly


def predict_file_access_anomaly(file_access_data):
    """Legacy function for file access anomaly prediction"""
    is_anomaly, score = detector.process_file_access({'features': file_access_data['features']})
    return is_anomaly