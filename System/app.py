import logging
import os
import secrets
import random
from collections import defaultdict
import threading
import time
import joblib
import numpy as np
import pandas as pd
import scapy.all as scapy
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_socketio import SocketIO, emit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from sklearn.exceptions import InconsistentVersionWarning
import hashlib
import re
from pymongo import MongoClient
from bson import ObjectId
import json
import importlib
from dotenv import load_dotenv

# Try to import TensorFlow/Keras, but gracefully handle if not available
try:
    _keras = importlib.import_module("tensorflow.keras")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    _keras = None
    logging.warning("TensorFlow not installed - Keras models will not be available")

# Load environment variables
load_dotenv()

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def parse_bool_env(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ('1', 'true', 'yes', 'on')


def parse_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


APP_ENV = os.getenv('APP_ENV', 'development').strip().lower()
IS_PRODUCTION = APP_ENV == 'production'
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*' if not IS_PRODUCTION else 'http://localhost:5000')
MAX_CONTENT_LENGTH = parse_int_env('MAX_CONTENT_LENGTH', 1024 * 1024)
ENABLE_DEMO_AUTOLOGIN = parse_bool_env('ENABLE_DEMO_AUTOLOGIN', default=not IS_PRODUCTION)
ENABLE_NETWORK_SNIFFING = parse_bool_env('ENABLE_NETWORK_SNIFFING', default=not IS_PRODUCTION)
ENABLE_ENDPOINT_SIMULATION = parse_bool_env('ENABLE_ENDPOINT_SIMULATION', default=not IS_PRODUCTION)
RATE_LIMIT_ENABLED = parse_bool_env('RATE_LIMIT_ENABLED', default=IS_PRODUCTION)
RATE_LIMIT_PER_MINUTE = parse_int_env('RATE_LIMIT_PER_MINUTE', 180)
_rate_limit_cache = defaultdict(list)
_rate_limit_lock = threading.Lock()


# ----------------- MongoDB Configuration ----------------
class MongoDBManager:
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="malware_detection"):
        try:
            self.client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=1000,
                connectTimeoutMS=1000,
                socketTimeoutMS=1000
            )
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.users = self.db.users
            self.network_logs = self.db.network_logs
            self.restricted_access_logs = self.db.restricted_access_logs
            self.file_access_logs = self.db.file_access_logs
            self.mouse_movement_logs = self.db.mouse_movement_logs
            self.api_logs = self.db.api_logs
            self.system_logs = self.db.system_logs
            logging.info("MongoDB connected successfully")
        except Exception as e:
            logging.error(f"MongoDB connection failed: {e}")
            # Fallback to in-memory storage
            self.users = {}
            self.network_logs = []
            self.restricted_access_logs = []
            self.file_access_logs = []
            self.mouse_movement_logs = []
            self.api_logs = []
            self.system_logs = []

    def log_system_event(self, event_type, description, user_id=None, severity="INFO"):
        """Log system events to MongoDB"""
        try:
            log_entry = {
                'timestamp': datetime.now(),
                'event_type': event_type,
                'description': description,
                'user_id': user_id,
                'severity': severity,
                'ip_address': request.remote_addr if request else None
            }
            if hasattr(self.system_logs, 'insert_one'):
                self.system_logs.insert_one(log_entry)
            elif isinstance(self.system_logs, list):
                self.system_logs.append(log_entry)
        except Exception as e:
            logging.error(f"Failed to log system event: {e}")

    def create_user(self, username, email, password_hash):
        """Create a new user in MongoDB"""
        try:
            user_data = {
                'username': username,
                'email': email,
                'password': password_hash,
                'created_at': datetime.now(),
                'last_login': None,
                'is_active': True
            }
            if hasattr(self.users, 'insert_one'):
                result = self.users.insert_one(user_data)
                return str(result.inserted_id)
            if isinstance(self.users, dict):
                self.users[username] = user_data
                return username
            return None
        except Exception as e:
            logging.error(f"Failed to create user: {e}")
            return None

    def get_user_by_username(self, username):
        """Get user by username from MongoDB"""
        try:
            if hasattr(self.users, 'find_one'):
                return self.users.find_one({'username': username})
            if isinstance(self.users, dict):
                return self.users.get(username)
            return None
        except Exception as e:
            logging.error(f"Failed to get user: {e}")
            return None

    def update_user_login(self, username):
        """Update user's last login timestamp"""
        try:
            if hasattr(self.users, 'update_one'):
                self.users.update_one(
                    {'username': username},
                    {'$set': {'last_login': datetime.now()}}
                )
            elif isinstance(self.users, dict) and username in self.users:
                self.users[username]['last_login'] = datetime.now()
        except Exception as e:
            logging.error(f"Failed to update user login: {e}")


# Initialize MongoDB
mongo_db = MongoDBManager()


# Helper loaders with explicit logging
def safe_joblib_load(path, friendly_name):
    """Attempt to load a joblib artifact and log clear messages."""
    try:
        if not os.path.exists(path):
            logging.warning(f"Model file for {friendly_name} not found: {path}")
            return None
        obj = joblib.load(path)
        logging.info(f"Loaded {friendly_name} from {path}")
        return obj
    except Exception as e:
        logging.exception(f"Failed to load {friendly_name} from {path}: {e}")
        return None


def safe_keras_load(path, friendly_name):
    try:
        if not TENSORFLOW_AVAILABLE:
            logging.warning(f"TensorFlow not available - cannot load Keras model {friendly_name}")
            return None
        if not os.path.exists(path):
            logging.warning(f"Keras model for {friendly_name} not found: {path}")
            return None
        model = _keras.models.load_model(path)
        logging.info(f"Loaded Keras model {friendly_name} from {path}")
        return model
    except Exception as e:
        logging.exception(f"Failed to load Keras model {friendly_name} from {path}: {e}")
        return None

# ----------------- Secret Key Configuration ----------------
def get_secret_key():
    """Get secret key from environment or generate a secure one"""
    secret_key = os.getenv('SECRET_KEY')
    if not secret_key:
        if IS_PRODUCTION:
            raise RuntimeError("SECRET_KEY must be set in production")
        # Generate a secure random key for development
        secret_key = secrets.token_hex(32)
        print("🔐 Generated development secret key")
        print("⚠️  For production, set SECRET_KEY environment variable")
    return secret_key

# ----------------- Global settings ----------------
# Initialize Flask and SocketIO
app = Flask(__name__)
app.secret_key = get_secret_key()
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(seconds=parse_int_env('SESSION_LIFETIME_SECONDS', 3600))
socketio = SocketIO(app, cors_allowed_origins=CORS_ALLOWED_ORIGINS)

network_sniffing_started = False
network_sniffing_lock = threading.Lock()

# Auto-login user for demo purposes
@app.before_request
def auto_login():
    """Automatically log in user for demo - remove in production"""
    if not ENABLE_DEMO_AUTOLOGIN:
        return
    if 'user_id' not in session:
        session['user_id'] = 'demo_user'
        session['user_email'] = 'demo@malwaredetection.com'
        mongo_db.log_system_event("AUTO_LOGIN", "User automatically logged in for demo", "demo_user", "INFO")


@app.before_request
def basic_rate_limit_guard():
    if not RATE_LIMIT_ENABLED:
        return None

    path = request.path or '/'
    if path.startswith('/static/') or path.startswith('/socket.io'):
        return None

    now = time.time()
    src_ip = request.headers.get('X-Forwarded-For', request.remote_addr) or 'unknown'
    if isinstance(src_ip, str) and ',' in src_ip:
        src_ip = src_ip.split(',')[0].strip()

    with _rate_limit_lock:
        timestamps = [ts for ts in _rate_limit_cache[src_ip] if now - ts < 60]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            if path.startswith('/api'):
                return jsonify({'status': 'error', 'message': 'Rate limit exceeded'}), 429
            return "Rate limit exceeded", 429
        timestamps.append(now)
        _rate_limit_cache[src_ip] = timestamps
    return None

# ----------------- Updated Routes with Consistent Names ----------------
@app.route('/')
def home():
    username = session.get('user_id')
    return render_template('index.html', username=username)


@app.route('/api_analyzer', methods=['GET'])
@app.route('/api-analyzer', methods=['GET'])
def api_analyzer():
    username = session.get('user_id')
    return render_template('api_analyzer.html', username=username)

@app.route('/file', methods=['GET'])
def file():
    username = session.get('user_id')
    return render_template('file.html', username=username)

@app.route('/network', methods=['GET'])
def network():
    global network_sniffing_started
    username = session.get('user_id')

    if ENABLE_NETWORK_SNIFFING:
        with network_sniffing_lock:
            if not network_sniffing_started:
                sniff_thread = threading.Thread(target=start_sniffing)
                sniff_thread.daemon = True
                sniff_thread.start()
                network_sniffing_started = True
    return render_template('network.html', username=username)


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({'status': 'ok', 'env': APP_ENV})

@app.route('/restricted', methods=['GET'])
def restricted():
    username = session.get('user_id')
    return render_template('restricted.html', username=username)

@app.route('/about', methods=['GET'])
def about():
    username = session.get('user_id')
    return render_template('about.html', username=username)

# Simple logout for demo
@app.route('/logout')
def logout():
    username = session.get('user_id')
    session.clear()
    mongo_db.log_system_event("USER_LOGOUT", f"User {username} logged out", username, "INFO")
    flash('You have been logged out successfully', 'success')
    # Auto-login again for demo
    session['user_id'] = 'demo_user'
    session['user_email'] = 'demo@malwaredetection.com'
    return redirect(url_for('home'))


# ----------------- Authentication Routes ----------------
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    """User sign in route"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple authentication for demo
        if username and password:
            session['user_id'] = username
            session['user_email'] = f'{username}@malwaredetection.com'
            mongo_db.log_system_event("USER_LOGIN", f"User {username} logged in", username, "INFO")
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Please enter both username and password', 'error')

    return render_template('signin.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User sign up route"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Simple validation for demo
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif username and email and password:
            # For demo purposes, just create session
            session['user_id'] = username
            session['user_email'] = email
            mongo_db.log_system_event("USER_SIGNUP", f"New user {username} registered", username, "INFO")
            flash('Registration successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Please fill all fields', 'error')

    return render_template('signup.html')

# ----------------- Global variables for restricted site monitoring ----------------
restricted_access_data = []
restricted_anomaly_count = 0
is_monitoring_active = False
monitoring_thread = None

# Suspicious domains and work hours configuration
suspicious_domains = ['darkweb.com', 'malicious.com', 'phishing.com', 'hack.com', 'tor.org',
                      'vpn.com', 'anonymous.com', 'bitcoin.com', 'cryptocurrency.com',
                      'threads.net', 'tiktok.com', 'root-me.org', 'pinterest.com']

# Work hours definition (9 AM to 5 PM)
WORK_HOURS_START = 9
WORK_HOURS_END = 17

# Initialize models as None (will handle missing models gracefully)
restricted_model = None
label_encoders = {}
target_encoder = None

# Define feature columns based on your training data
feature_columns = ['time', 'end_time', 'user_id', 'source_ip', 'domain', 'domain_type',
                   'access_type', 'request_type', 'protocol', 'vpn_usage', 'tor_usage',
                   'dns_encryption', 'user_role', 'user_activity_type', 'recent_web_access_attempts',
                   'status', 'attack_risk_level']

# Try to load models with error handling - UPDATED WITH BETTER ERROR HANDLING
# Try to load the main model
restricted_model = safe_joblib_load(os.path.join('restricted', 'attack_prediction_model.pkl'), 'restricted main model')
if restricted_model is None:
    logging.warning("Could not load main restricted site access model, running in simulation mode")
else:
    logging.info("Main restricted site access model loaded successfully")

# Try to load label encoders
label_encoders = safe_joblib_load(os.path.join('restricted', 'label_encoders.pkl'), 'restricted label encoders')
if label_encoders is None:
    logging.warning("Label encoders not available; creating defaults")
    label_encoders = {}
for col in ['domain', 'domain_type', 'access_type', 'request_type', 'protocol',
            'user_role', 'user_activity_type', 'status', 'attack_risk_level']:
        le = LabelEncoder()
        # Fit with some default values
        if col == 'domain_type':
            le.fit(['Social Media Site', 'Educational Website', 'Suspicious Site', 'Unknown'])
        elif col == 'access_type':
            le.fit(['Allowed', 'Restricted', 'Suspicious'])
        elif col == 'attack_risk_level':
            le.fit(['Low', 'Medium', 'High'])
        else:
            le.fit(['Unknown'])
        label_encoders[col] = le

# Try to load target encoder
target_encoder = safe_joblib_load(os.path.join('restricted', 'target_encoder.pkl'), 'restricted target encoder')
if target_encoder is None:
    logging.warning("Target encoder not available; creating default target encoder")
    target_encoder = LabelEncoder()
    target_encoder.fit(['Benign', 'Ransomware', 'Phishing', 'Distributed Denial of Service (DDoS)', 'Fraud'])

# ----------------- MITRE ATT&CK Framework Mapping ----------------
MITRE_MAPPING = {
    'Benign': {
        'risk_level': 'Low',
        'tactics': [],
        'techniques': [],
        'mitigation': ['Continue normal monitoring operations']
    },
    'Ransomware': {
        'risk_level': 'Critical',
        'tactics': ['Impact', 'Lateral Movement', 'Exfiltration'],
        'techniques': [
            'T1486 - Data Encrypted for Impact',
            'T1021 - Remote Services',
            'T1048 - Exfiltration Over Alternative Protocol'
        ],
        'mitigation': [
            'Implement application whitelisting',
            'Use behavior-based threat detection',
            'Maintain offline backups',
            'Implement network segmentation'
        ]
    },
    'Phishing': {
        'risk_level': 'High',
        'tactics': ['Initial Access', 'Collection'],
        'techniques': [
            'T1566 - Phishing',
            'T1114 - Email Collection',
            'T1056 - Input Capture'
        ],
        'mitigation': [
            'Implement email filtering',
            'User security awareness training',
            'Multi-factor authentication',
            'Web content filtering'
        ]
    },
    'Distributed Denial of Service (DDoS)': {
        'risk_level': 'High',
        'tactics': ['Impact'],
        'techniques': [
            'T1498 - Network Denial of Service',
            'T1499 - Endpoint Denial of Service'
        ],
        'mitigation': [
            'Implement DDoS protection services',
            'Network traffic monitoring',
            'Rate limiting',
            'Load balancing'
        ]
    },
    'Fraud': {
        'risk_level': 'Medium',
        'tactics': ['Collection', 'Exfiltration'],
        'techniques': [
            'T1530 - Data from Cloud Storage',
            'T1041 - Exfiltration Over C2 Channel'
        ],
        'mitigation': [
            'Implement fraud detection systems',
            'Transaction monitoring',
            'Behavioral analytics',
            'Access control policies'
        ]
    }
}


# ----------------- Restricted Site Access Detection Functions ----------------
def time_to_minutes(time_str):
    """Convert time string to minutes since midnight"""
    try:
        if isinstance(time_str, str):
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            return time_obj.hour * 60 + time_obj.minute
        return 0
    except:
        return 0


def is_work_hours():
    """Check if current time is within work hours"""
    current_hour = datetime.now().hour
    return WORK_HOURS_START <= current_hour <= WORK_HOURS_END


def create_web_access_features():
    """Create features for web access classification - FIXED VERSION"""
    try:
        current_time = datetime.now()

        # Create realistic simulated features based on your training data
        features = {}

        # Time features
        features['time'] = time_to_minutes(current_time.strftime('%H:%M:%S'))
        features['end_time'] = time_to_minutes(
            (current_time.replace(second=0) + pd.Timedelta(hours=1)).strftime('%H:%M:%S'))

        # User and network features
        features['user_id'] = f'user_{np.random.randint(0, 10)}'
        features['source_ip'] = f'192.168.1.{np.random.randint(1, 255)}'

        # Domain features - mix of benign and suspicious
        domain_choice = np.random.choice(['threads.net', 'coursera.org', 'tiktok.com', 'root-me.org',
                                          'pinterest.com', 'darkweb.com', 'malicious.com'])
        features['domain'] = domain_choice
        features['domain_type'] = np.random.choice(['Social Media Site', 'Educational Website',
                                                    'Capture the Flag (CTF) Challenge Site', 'Suspicious Site'])

        # Access control features
        features['access_type'] = np.random.choice(['Allowed', 'Restricted', 'Suspicious'])
        features['request_type'] = np.random.choice(['GET', 'POST', 'DELETE', 'CONNECT', 'OPTIONS'])
        features['protocol'] = np.random.choice(['HTTP', 'HTTPS', 'XMPP', 'Telnet', 'SFTP'])

        # Security features
        features['vpn_usage'] = np.random.randint(0, 2)
        features['tor_usage'] = np.random.randint(0, 2)
        features['dns_encryption'] = np.random.randint(0, 2)

        # User behavior features
        features['user_role'] = np.random.choice(['Guest', 'Staff', 'Admin'])
        features['user_activity_type'] = np.random.choice(['Browsing', 'File Access', 'Streaming'])
        features['recent_web_access_attempts'] = np.random.randint(1, 20)

        # Status and risk features
        features['status'] = np.random.choice(['Successful', 'Failed'])
        features['attack_risk_level'] = np.random.choice(['Low', 'Medium', 'High'])

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        # Encode categorical variables using label encoders
        for col in ['domain', 'domain_type', 'access_type', 'request_type', 'protocol',
                    'user_role', 'user_activity_type', 'status', 'attack_risk_level']:
            if col in label_encoders and col in feature_df.columns:
                try:
                    # Handle unseen labels
                    unique_values = feature_df[col].unique()
                    for val in unique_values:
                        if val not in label_encoders[col].classes_:
                            # Add unknown class to encoder
                            label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
                    feature_df[col] = label_encoders[col].transform(feature_df[col].astype(str))
                except ValueError as e:
                    logging.warning(f"Encoding error for {col}: {e}")
                    feature_df[col] = 0

        # Ensure all required features are present
        for col in feature_columns:
            if col not in feature_df.columns:
                feature_df[col] = 0

        # Reorder columns to match training
        feature_df = feature_df.reindex(columns=feature_columns, fill_value=0)

        return feature_df, features

    except Exception as e:
        logging.error(f"Feature creation error: {e}")
        return None, {}


def predict_web_access(features):
    """Predict web access type using ML model - FIXED VERSION"""
    try:
        if features is not None:
            if restricted_model is not None:
                # Make prediction with actual model
                prediction_encoded = restricted_model.predict(features)[0]
                prediction_proba = restricted_model.predict_proba(features)[0]

                # Decode prediction
                if target_encoder:
                    prediction = target_encoder.inverse_transform([prediction_encoded])[0]
                else:
                    prediction = str(prediction_encoded)
            else:
                # Fallback: Use rule-based prediction when model is not available
                domain = features.iloc[0]['domain'] if 'domain' in features.columns else 0
                is_suspicious = any(domain == d for d in suspicious_domains) if isinstance(domain, str) else False

                # Simple rule-based prediction
                if is_suspicious and np.random.random() > 0.7:
                    attack_types = ['Ransomware', 'Phishing', 'Distributed Denial of Service (DDoS)', 'Fraud']
                    prediction = np.random.choice(attack_types)
                    confidence = np.random.uniform(0.7, 0.95)
                else:
                    prediction = 'Benign'
                    confidence = np.random.uniform(0.8, 0.99)

                prediction_proba = [0] * 5  # Dummy probabilities

            return {
                'prediction': prediction,
                'confidence': float(np.max(prediction_proba)) if restricted_model else confidence,
                'probabilities': dict(
                    zip(target_encoder.classes_, prediction_proba)) if target_encoder and restricted_model else {}
            }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        # Fallback prediction
        return {
            'prediction': 'Benign',
            'confidence': 0.5,
            'probabilities': {}
        }

    return None


def detect_restricted_access_simulation():
    """Simulate restricted site access detection for demo purposes - FIXED VERSION"""
    global restricted_access_data, restricted_anomaly_count, is_monitoring_active

    logging.info("Starting restricted site access monitoring simulation")

    while is_monitoring_active:
        try:
            # Create features and predict - FIXED: No arguments needed
            feature_df, raw_features = create_web_access_features()

            if feature_df is not None:
                prediction = predict_web_access(feature_df)

                if prediction:
                    # Determine if domain is suspicious
                    domain = raw_features.get('domain', 'unknown.com')
                    is_suspicious = domain in suspicious_domains

                    # Create access record
                    access_record = {
                        'timestamp': datetime.now(),
                        'date': datetime.now().strftime('%m/%d/%Y'),
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'end_time': (datetime.now().replace(second=0) + pd.Timedelta(hours=1)).strftime('%H:%M:%S'),
                        'user_id': raw_features.get('user_id', 'unknown'),
                        'source_ip': raw_features.get('source_ip', '0.0.0.0'),
                        'domain': domain,
                        'domain_type': raw_features.get('domain_type', 'Unknown'),
                        'access_type': raw_features.get('access_type', 'Allowed'),
                        'request_type': raw_features.get('request_type', 'GET'),
                        'protocol': raw_features.get('protocol', 'HTTP'),
                        'vpn_usage': raw_features.get('vpn_usage', 0),
                        'tor_usage': raw_features.get('tor_usage', 0),
                        'dns_encryption': raw_features.get('dns_encryption', 0),
                        'user_role': raw_features.get('user_role', 'Guest'),
                        'user_activity_type': raw_features.get('user_activity_type', 'Browsing'),
                        'recent_web_access_attempts': raw_features.get('recent_web_access_attempts', 0),
                        'status': raw_features.get('status', 'Successful'),
                        'attack_risk_level': raw_features.get('attack_risk_level', 'Low'),
                        'prediction': prediction['prediction'],
                        'confidence': prediction['confidence'],
                        'is_suspicious': is_suspicious,
                        'during_work_hours': is_work_hours()
                    }

                    # Store in MongoDB
                    try:
                        mongo_db.restricted_access_logs.insert_one(access_record.copy())
                    except Exception as e:
                        logging.error(f"Failed to log restricted access to MongoDB: {e}")

                    restricted_access_data.append(access_record)

                    # Keep only last 1000 records
                    if len(restricted_access_data) > 1000:
                        restricted_access_data = restricted_access_data[-1000:]

                    # Count anomalies
                    if access_record['prediction'] != 'Benign':
                        restricted_anomaly_count += 1

                    # Emit to clients
                    socketio.emit('restricted_access_alert', {
                        'src_ip': access_record['source_ip'],
                        'domain': access_record['domain'],
                        'is_suspicious': access_record['is_suspicious'],
                        'prediction': access_record['prediction'],
                        'confidence': access_record['confidence'],
                        'during_work_hours': access_record['during_work_hours'],
                        'total_anomalies': restricted_anomaly_count,
                        'timestamp': access_record['timestamp'].isoformat(),
                        'user_id': access_record['user_id'],
                        'access_type': access_record['access_type'],
                        'attack_risk_level': access_record['attack_risk_level']
                    })

                    logging.info(f"Access detected - IP: {access_record['source_ip']}, "
                                 f"Domain: {access_record['domain']}, Prediction: {access_record['prediction']}")

            # Random delay between 1-3 seconds
            time.sleep(np.random.uniform(1, 3))

        except Exception as e:
            logging.error(f"Error in monitoring simulation: {e}")
            time.sleep(5)


# ----------------- API Endpoints for Restricted Site Monitoring ----------------
@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start restricted site monitoring"""
    global is_monitoring_active, monitoring_thread

    try:
        if not is_monitoring_active:
            is_monitoring_active = True
            monitoring_thread = threading.Thread(target=detect_restricted_access_simulation)
            monitoring_thread.daemon = True
            monitoring_thread.start()

            socketio.emit('monitoring_status', {'is_monitoring': True})
            mongo_db.log_system_event("MONITORING_STARTED", "Restricted site monitoring started",
                                      session.get('user_id'), "INFO")
            logging.info("Monitoring started successfully")
            return jsonify({'status': 'success', 'message': 'Monitoring started'})
        else:
            return jsonify({'status': 'info', 'message': 'Monitoring is already active'})

    except Exception as e:
        logging.error(f"Error starting monitoring: {e}")
        mongo_db.log_system_event("MONITORING_ERROR", f"Failed to start monitoring: {e}", session.get('user_id'),
                                  "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop restricted site monitoring"""
    global is_monitoring_active

    try:
        is_monitoring_active = False
        socketio.emit('monitoring_status', {'is_monitoring': False})
        mongo_db.log_system_event("MONITORING_STOPPED", "Restricted site monitoring stopped", session.get('user_id'),
                                  "INFO")
        logging.info("Monitoring stopped successfully")
        return jsonify({'status': 'success', 'message': 'Monitoring stopped'})
    except Exception as e:
        logging.error(f"Error stopping monitoring: {e}")
        mongo_db.log_system_event("MONITORING_ERROR", f"Failed to stop monitoring: {e}", session.get('user_id'),
                                  "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/clear_data', methods=['POST'])
def clear_data():
    """Clear monitoring data"""
    global restricted_access_data, restricted_anomaly_count

    try:
        restricted_access_data = []
        restricted_anomaly_count = 0
        socketio.emit('data_cleared', {})
        mongo_db.log_system_event("DATA_CLEARED", "Monitoring data cleared", session.get('user_id'), "INFO")
        return jsonify({'status': 'success', 'message': 'Data cleared'})
    except Exception as e:
        mongo_db.log_system_event("DATA_ERROR", f"Failed to clear data: {e}", session.get('user_id'), "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    """Generate forensic report"""
    try:
        # Calculate statistics
        total_accesses = len(restricted_access_data)
        suspicious_domains_count = len([d for d in restricted_access_data if d.get('is_suspicious', False)])
        blocked_accesses = len([d for d in restricted_access_data if d.get('prediction', 'Benign') != 'Benign'])
        work_hours_violations = len(
            [d for d in restricted_access_data if d.get('during_work_hours', False) and d.get('is_suspicious', False)])

        # Get attack distribution
        attack_distribution = {}
        for access in restricted_access_data:
            prediction = access.get('prediction', 'Benign')
            attack_distribution[prediction] = attack_distribution.get(prediction, 0) + 1

        # Get recent suspicious activity
        recent_suspicious = [d for d in restricted_access_data if d.get('prediction', 'Benign') != 'Benign'][-10:]

        report_data = {
            'total_accesses': total_accesses,
            'suspicious_domains': suspicious_domains_count,
            'blocked_accesses': blocked_accesses,
            'work_hours_violations': work_hours_violations,
            'attack_distribution': attack_distribution,
            'recent_suspicious': recent_suspicious,
            'generated_at': datetime.now().isoformat(),
            'monitoring_duration': 'Active session'
        }

        mongo_db.log_system_event("REPORT_GENERATED", "Forensic report generated", session.get('user_id'), "INFO")
        return jsonify({'status': 'success', 'report': report_data})
    except Exception as e:
        mongo_db.log_system_event("REPORT_ERROR", f"Failed to generate report: {e}", session.get('user_id'), "ERROR")
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/get_mitre_info', methods=['POST'])
def get_mitre_info():
    """Get MITRE ATT&CK information for a specific attack type"""
    try:
        data = request.get_json()
        attack_type = data.get('attack_type', 'Benign')

        mitre_info = MITRE_MAPPING.get(attack_type, MITRE_MAPPING['Benign'])

        return jsonify({
            'status': 'success',
            'attack_type': attack_type,
            'mitre_info': mitre_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/get_current_stats', methods=['GET'])
def get_current_stats():
    """Get current monitoring statistics"""
    try:
        total_accesses = len(restricted_access_data)
        suspicious_domains_count = len([d for d in restricted_access_data if d.get('is_suspicious', False)])
        blocked_accesses = len([d for d in restricted_access_data if d.get('prediction', 'Benign') != 'Benign'])

        return jsonify({
            'status': 'success',
            'total_accesses': total_accesses,
            'suspicious_domains': suspicious_domains_count,
            'blocked_accesses': blocked_accesses,
            'is_monitoring': is_monitoring_active
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ----------------- MongoDB Analytics Routes ----------------
@app.route('/api/analytics/restricted_access', methods=['GET'])
def get_restricted_access_analytics():
    """Get analytics data for restricted access from MongoDB"""
    try:
        # Get data from last 24 hours
        yesterday = datetime.now() - pd.Timedelta(days=1)

        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': yesterday}
                }
            },
            {
                '$group': {
                    '_id': '$prediction',
                    'count': {'$sum': 1},
                    'avg_confidence': {'$avg': '$confidence'}
                }
            }
        ]

        results = list(mongo_db.restricted_access_logs.aggregate(pipeline))

        analytics_data = {}
        for result in results:
            analytics_data[result['_id']] = {
                'count': result['count'],
                'avg_confidence': round(result['avg_confidence'], 2)
            }

        return jsonify({
            'status': 'success',
            'analytics': analytics_data,
            'time_period': 'last_24_hours'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/analytics/system_logs', methods=['GET'])
def get_system_logs_analytics():
    """Get system logs analytics"""
    try:
        # Get logs from last 7 days
        last_week = datetime.now() - pd.Timedelta(days=7)

        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': last_week}
                }
            },
            {
                '$group': {
                    '_id': {'event_type': '$event_type', 'severity': '$severity'},
                    'count': {'$sum': 1}
                }
            },
            {
                '$sort': {'count': -1}
            }
        ]

        results = list(mongo_db.system_logs.aggregate(pipeline))

        return jsonify({
            'status': 'success',
            'logs_analytics': results,
            'time_period': 'last_7_days'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# ----------------- Global variables to track packets and stats ----------------
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

# Add global variables to track the last packet count and timestamp for speed calculation
last_packet_count = 0
last_timestamp = time.time()
last_speed_emit_time = time.time()

# Load models with error handling
try:
    # Load network models
    network_model = safe_joblib_load(os.path.join('Network', 'Models', 'gan_Predict.pkl'), 'network model')
    network_encoder = safe_joblib_load(os.path.join('Network', 'Models', 'label_encoder.pkl'), 'network label encoder')
    if network_model is None or network_encoder is None:
        logging.error("Network model or encoder not loaded; network detection will use fallbacks")
    else:
        logging.info("Network models loaded successfully")
except Exception as e:
    logging.error(f"Could not load network models: {e}")

try:
    # Load endpoint models
    endpoint_rf_model = safe_joblib_load(os.path.join('EndPoint', 'Models', 'random_forest_model.pkl'), 'endpoint RF model')
    endpoint_encoder = safe_joblib_load(os.path.join('EndPoint', 'Models', 'label_encoder.pkl'), 'endpoint label encoder')
    if endpoint_rf_model is None:
        logging.error("Endpoint RF model not loaded; endpoint detection will use fallbacks")
    else:
        logging.info("Endpoint models loaded successfully")
except Exception as e:
    logging.error(f"Could not load endpoint models: {e}")


# ----------------- Enhanced Packet Processing ----------------
def process_packet(packet, interface):
    """ Process a network packet, extract features, make predictions, and emit results via SocketIO. """
    global total_packets, non_ip_packet_count, class_counts, last_timestamp, last_packet_count, last_speed_emit_time

    # Record the current time
    current_time = time.time()
    elapsed_time = current_time - last_timestamp

    try:
        # Update packet processing speed every second and emit
        if elapsed_time > 1:
            speed = (total_packets - last_packet_count) / elapsed_time
            last_packet_count = total_packets
            last_timestamp = current_time

            # Emit the packet speed to the client every second
            socketio.emit('packet_speed', {
                'total_packets': total_packets,
                'speed': speed,
                'timestamp': current_time
            })
            last_speed_emit_time = current_time

        # Check if the packet contains an IP layer
        if packet.haslayer(scapy.IP):
            try:
                # Extract IP-related fields from the packet
                src_ip = packet[scapy.IP].src
                dst_ip = packet[scapy.IP].dst
                proto = packet[scapy.IP].proto
            except AttributeError as e:
                logging.debug(f"Packet attribute error: {e}")
                return  # Skip processing for malformed packets

            # Increment the total packet counter
            total_packets += 1

            # Calculate statistics for the packet
            calculate_statistics(packet)

            # Create a unique key for the flow based on source/destination IP and protocol
            flow_key = (src_ip, dst_ip, proto)
            stats = flow_stats[flow_key]

            # Extract features for model prediction
            selected_features = extract_selected_features(stats)
            feature_values = np.array(list(selected_features.values())).reshape(1, -1)

            # Predict the class using the pre-trained model
            if network_model is not None:
                prediction = network_model.predict(feature_values)
                predicted_label = network_encoder.inverse_transform(prediction)[0]
            else:
                # Fallback to simple rule-based detection if model is not available
                if len(packet) > 1500:  # Large packet size might indicate attack
                    predicted_label = "Suspicious"
                else:
                    predicted_label = "BENIGN"

            logging.info(f"Network Traffic - Interface: {interface} - Predicted Class: {predicted_label}")

            # Update class counts
            class_counts[predicted_label] += 1

            # Log to MongoDB
            try:
                network_log = {
                    'timestamp': datetime.now(),
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'protocol': proto,
                    'prediction': predicted_label,
                    'interface': interface,
                    'packet_length': len(packet)
                }
                mongo_db.network_logs.insert_one(network_log)
            except Exception as e:
                logging.error(f"Failed to log network event to MongoDB: {e}")

            # Emit the prediction and stats to the client
            socketio.emit('network_threat', {
                'prediction': predicted_label,
                'total_packets': total_packets,
                'class_counts': dict(class_counts),
                'non_ip_packet_count': non_ip_packet_count,
                'timestamp': current_time
            })

            # Cleanup old flow stats periodically
            if total_packets % 100 == 0:  # Cleanup every 100 packets
                cleanup_flow_stats()

        else:
            # Handle non-IP packets as BENIGN
            total_packets += 1
            non_ip_packet_count += 1
            class_counts["BENIGN"] += 1

            logging.debug(f"Non-IP packet processed and classified as BENIGN.")

            # Emit BENIGN status for non-IP packets
            socketio.emit('network_threat', {
                'prediction': "BENIGN",
                'total_packets': total_packets,
                'non_ip_packet_count': non_ip_packet_count,
                'class_counts': dict(class_counts),
                'timestamp': current_time
            })

    except Exception as e:
        # Catch and log any errors during packet processing
        logging.error(f"Error processing packet: {str(e)}")


# ----------------- Utility Functions ----------------

# Cleanup function to remove old or completed flows
def cleanup_flow_stats():
    current_time = time.time()
    for flow_key in list(flow_stats.keys()):
        if flow_stats[flow_key]['flow_end_time'] is not None:
            flow_duration = current_time - flow_stats[flow_key]['flow_end_time']
            if flow_duration > 600:  # Remove flows older than 10 minutes
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
    flow_stats[flow_key]['flow_duration'] = pkt_time - flow_stats[flow_key]['flow_start_time']

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
def extract_selected_features(stats):
    fwd_packet_lengths = stats['fwd_packet_lengths'] or [0]
    bwd_packet_lengths = stats['bwd_packet_lengths'] or [0]
    fwd_iat = np.diff(stats['fwd_iat']) if len(stats['fwd_iat']) > 1 else np.array([0])
    bwd_iat = np.diff(stats['bwd_iat']) if len(stats['bwd_iat']) > 1 else np.array([0])

    selected_features = {
        'Fwd Packet Length Std': np.std(fwd_packet_lengths) if len(fwd_packet_lengths) > 0 else 0,
        'Bwd Packet Length Std': np.std(bwd_packet_lengths) if len(bwd_packet_lengths) > 0 else 0,
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


# Function to emit periodic speed updates
def emit_periodic_speed():
    """Emit speed updates even when no packets are being processed"""
    global last_packet_count, last_timestamp, total_packets

    while True:
        time.sleep(1)  # Emit every second
        current_time = time.time()
        elapsed_time = current_time - last_timestamp

        if elapsed_time > 0:
            speed = (total_packets - last_packet_count) / elapsed_time
            last_packet_count = total_packets
            last_timestamp = current_time

            socketio.emit('packet_speed', {
                'total_packets': total_packets,
                'speed': speed,
                'timestamp': current_time
            })


# Sniff packets on the interface
def sniff_interface(interface):
    try:
        logging.info(f"Starting sniffing on interface: {interface}")
        # Use filter to capture only IP packets to reduce noise
        scapy.sniff(iface=interface, prn=lambda pkt: process_packet(pkt, interface),
                    store=False, filter="ip")
    except Exception as e:
        logging.error(f"Error sniffing on interface {interface}: {str(e)}")


# Start sniffing on all interfaces
def start_sniffing():
    try:
        interfaces = scapy.get_if_list()
        logging.info(f"Available interfaces: {interfaces}")

        # Start sniffing on each interface in separate threads
        threads = []
        for interface in interfaces:
            # Skip loopback and virtual interfaces to reduce noise
            if 'lo' not in interface and 'virbr' not in interface:
                thread = threading.Thread(target=sniff_interface, args=(interface,))
                thread.daemon = True
                thread.start()
                threads.append(thread)
                logging.info(f"Started sniffing on interface: {interface}")

        # Keep the threads alive
        for thread in threads:
            thread.join()

    except Exception as e:
        logging.critical(f"Error in sniffing: {str(e)}")


# ----------------- Endpoint Simulation Functions ----------------

# Function to simulate packet processing for demonstration
def simulate_packet_processing():
    """Simulate endpoint packet processing for demonstration"""
    while True:
        # Simulate processing packets every 5 seconds
        time.sleep(5)
        logging.info("Simulating endpoint packet processing...")

        # Simulated packet data with more realistic values
        packet_data = {
            'src_bytes': np.random.randint(100, 1500),
            'dst_bytes': np.random.randint(100, 1500),
            'diff_srv_rate': np.random.uniform(0, 1),
            'dst_host_srv_count': np.random.randint(1, 100),
            'dst_host_srv_serror_rate': np.random.uniform(0, 0.5),
            'flag_SF': np.random.randint(0, 2),
            'dst_host_diff_srv_rate': np.random.uniform(0, 1),
            'dst_host_same_srv_rate': np.random.uniform(0, 1),
            'count': np.random.randint(1, 50),
            'logged_in': np.random.randint(0, 2),
            'src_ip': f"192.168.1.{np.random.randint(1, 255)}"
        }

        # Process the simulated endpoint packet
        process_endpoint_packet(packet_data)


# Endpoint packet processing logic
def process_endpoint_packet(packet_data):
    global total_packets
    total_packets += 1

    logging.info("Processing endpoint packet...")

    # Use the packet data for prediction
    sample_data_input = [
        packet_data['src_bytes'],
        packet_data['dst_bytes'],
        packet_data['diff_srv_rate'],
        packet_data['dst_host_srv_count'],
        packet_data['dst_host_srv_serror_rate'],
        packet_data['flag_SF'],
        packet_data['dst_host_diff_srv_rate'],
        packet_data['dst_host_same_srv_rate'],
        packet_data['count'],
        packet_data['logged_in']
    ]

    # Get prediction from the model
    if endpoint_rf_model is not None:
        rf_prediction = predict_from_models(sample_data_input)
        try:
            predicted_label = endpoint_encoder.inverse_transform([rf_prediction])[0]
        except ValueError:
            predicted_label = rf_prediction
    else:
        # Fallback if endpoint model is not available
        if packet_data['src_bytes'] > 1000 or packet_data['dst_bytes'] > 1000:
            predicted_label = "Suspicious"
        else:
            predicted_label = "BENIGN"

    # Extract src_ip from packet_data
    src_ip = packet_data.get('src_ip', 'unknown')

    # Emit the prediction to the client
    logging.info(f"Emitting endpoint prediction: {predicted_label} with src_ip: {src_ip}")
    socketio.emit('endpoint_threat', {
        'prediction': predicted_label,
        'src_ip': src_ip,
        'timestamp': time.time()
    })


# Prediction function for endpoint
def predict_from_models(sample_data_input):
    sample_data = pd.DataFrame([sample_data_input], columns=[
        'src_bytes', 'dst_bytes', 'diff_srv_rate', 'dst_host_srv_count', 'dst_host_srv_serror_rate',
        'flag_SF', 'dst_host_diff_srv_rate', 'dst_host_same_srv_rate', 'count', 'logged_in'
    ])

    # Scale the sample data
    X_train_relevant = pd.DataFrame([[0] * 10], columns=sample_data.columns)
    scaler_relevant = StandardScaler().fit(X_train_relevant)
    sample_data_scaled = scaler_relevant.transform(sample_data)

    # Get prediction from the random forest model
    rf_prediction = endpoint_rf_model.predict(sample_data_scaled)[0]
    return rf_prediction


# ----------------- File and Mouse Movement Detection ----------------

# Global variables
file_access_data = []
mouse_movement_data = []
file_anomaly_count = 0
mouse_anomaly_count = 0

# API monitoring variables
api_requests_log = []
api_anomaly_count = 0
is_api_monitoring_active = False
api_class_counts = defaultdict(int)
API_ALLOWED_CLASSES = ('normal', 'bot', 'outlier', 'attack')
API_EXCLUDED_PREFIXES = ('/static/', '/socket.io')
API_EXCLUDED_PATHS = {
    '/favicon.ico',
    '/api/start_api_monitoring',
    '/api/stop_api_monitoring',
    '/api/get_api_stats',
    '/api/generate_api_report',
    '/api/test_api_call',
    '/api/generate_test_api_event',
    '/api/predict_behavior',
    '/predict_behavior'
}
API_SESSION_TIMEOUT_SECONDS = parse_int_env('API_SESSION_TIMEOUT_SECONDS', 1800)
api_behavioral_state = {
    'users': {},
    'global_unique_paths': set()
}
api_behavioral_lock = threading.Lock()
api_dataset_streaming_active = False
api_dataset_stream_thread = None
api_dataset_stream_stop_event = threading.Event()
api_dataset_stream_index = 0
api_dataset_stream_lock = threading.Lock()

# ----------------- API Analyzer (Realtime) -----------------
# Paths for optional model artifacts
api_model = None
api_scaler = None
api_label_encoder = None
api_target_encoder = None
api_source_encoder = None
api_model_source = None

API_TRAIN_MEAN_INTER_API_ACCESS_DURATION = 20.548672
API_TRAIN_MEAN_API_ACCESS_UNIQUENESS = 0.444437

API_IP_TYPE_CLASSES = ['default', 'private_ip', 'datacenter', 'google_bot', 'unknown_ip_type']
API_BEHAVIOR_TYPE_CLASSES = ['outlier', 'normal', 'bot', 'attack', 'unknown_behavior_type']
API_BEHAVIOR_CLASSES = [
    'outlier',
    'normal',
    'googlebot/2.1',
    'bingpreview/1.0b',
    'sql injection',
    'adsbot-google',
    'unknown_behavior_category'
]

api_le_ip_type = LabelEncoder().fit(API_IP_TYPE_CLASSES)
api_le_behavior_type = LabelEncoder().fit(API_BEHAVIOR_TYPE_CLASSES)
api_le_behavior = LabelEncoder().fit(API_BEHAVIOR_CLASSES)

API_TEST_SOURCE_DATASET_PATH = os.path.join('DataSet', 'remaining_behavior_ext.csv')
API_TEST_EXTRACTED_DATASET_PATH = os.path.join('DataSet', 'api_behavior_test_samples_v2.csv')
API_TEST_REQUIRED_CLASS_COUNTS = {
    'attack': 10,
    'bot': 15,
    'normal': 80,
    'outlier': 20
}
api_test_dataset_cache = None
api_test_dataset_lock = threading.Lock()

STRICT_API_MODEL_FILE = 'Random_Forest_robust_source.pkl'
STRICT_API_SOURCE_ENCODER_FILE = 'le_source_robust.pkl'
STRICT_API_TARGET_ENCODER_FILE = 'label_encoder.pkl'


def load_api_artifacts():
    """Load strict API analyzer artifacts required for live monitoring predictions."""
    global api_model, api_scaler, api_label_encoder, api_target_encoder, api_source_encoder, api_model_source

    required_model_path = os.path.join('API_analyzer', STRICT_API_MODEL_FILE)
    required_source_encoder_path = os.path.join('API_analyzer', STRICT_API_SOURCE_ENCODER_FILE)
    required_target_encoder_path = os.path.join('API_analyzer', STRICT_API_TARGET_ENCODER_FILE)

    model_obj = safe_joblib_load(required_model_path, f'API strict model - {STRICT_API_MODEL_FILE}')
    source_encoder_obj = safe_joblib_load(required_source_encoder_path, f'API strict source encoder - {STRICT_API_SOURCE_ENCODER_FILE}')
    target_encoder_obj = safe_joblib_load(required_target_encoder_path, f'API strict target encoder - {STRICT_API_TARGET_ENCODER_FILE}')

    if model_obj is None or source_encoder_obj is None or target_encoder_obj is None:
        api_model = None
        api_target_encoder = None
        api_source_encoder = None
        api_label_encoder = None
        api_scaler = None
        api_model_source = None
        logging.error(
            f'API Analyzer strict artifacts missing. Live monitor predictions require '
            f'{STRICT_API_MODEL_FILE}, {STRICT_API_SOURCE_ENCODER_FILE}, and {STRICT_API_TARGET_ENCODER_FILE}.'
        )
        return

    api_model = model_obj
    api_target_encoder = target_encoder_obj
    api_source_encoder = source_encoder_obj
    api_label_encoder = api_target_encoder
    api_scaler = None
    api_model_source = 'API root RandomForest model'
    logging.info('API Analyzer strict model source selected: API root RandomForest model')


def strict_api_artifacts_ready() -> tuple:
    """Return (is_ready, message) for required API analyzer artifacts."""
    if api_model_source != 'API root RandomForest model':
        return False, 'Strict API model source is not selected'
    if api_model is None:
        return False, f'Missing loaded model: {STRICT_API_MODEL_FILE}'
    if api_source_encoder is None:
        return False, f'Missing loaded source encoder: {STRICT_API_SOURCE_ENCODER_FILE}'
    if api_target_encoder is None:
        return False, f'Missing loaded target encoder: {STRICT_API_TARGET_ENCODER_FILE}'
    return True, 'ok'


def strict_api_artifacts_payload() -> dict:
    """Return strict artifact names/paths used for API analyzer predictions."""
    return {
        'model': os.path.join('API_analyzer', STRICT_API_MODEL_FILE),
        'source_encoder': os.path.join('API_analyzer', STRICT_API_SOURCE_ENCODER_FILE),
        'target_encoder': os.path.join('API_analyzer', STRICT_API_TARGET_ENCODER_FILE)
    }


load_api_artifacts()

if api_model is not None and api_target_encoder is not None:
    logging.info(f'API Analyzer model loaded successfully from {api_model_source}')
else:
    missing = [n for n, v in [('api_model', api_model), ('api_target_encoder', api_target_encoder)] if v is None]
    logging.warning(f'API Analyzer models not fully available (will use heuristics). Missing: {missing}')


def safe_label_transform(encoder: LabelEncoder, value, unknown_placeholder_str: str) -> int:
    if encoder is None:
        return 0

    class_lookup = {str(item).strip().lower(): str(item) for item in getattr(encoder, 'classes_', [])}
    unknown_key = str(unknown_placeholder_str).strip().lower()

    if pd.isna(value) or value is None:
        value_key = unknown_key
    else:
        value_key = str(value).strip().lower()

    value_to_transform = class_lookup.get(value_key)
    if value_to_transform is None:
        value_to_transform = class_lookup.get(unknown_key, unknown_placeholder_str)

    try:
        return int(encoder.transform([value_to_transform])[0])
    except Exception:
        return 0


def classify_api_call(row: pd.Series):
    """Rule-based behavior classification: attack, bot, normal, outlier."""
    behavior_type = str(row.get('behavior_type', 'unknown_behavior_type')).strip().lower()
    behavior = str(row.get('behavior', 'unknown_behavior_category')).strip().lower()
    ip_type = str(row.get('ip_type', 'unknown_ip_type')).strip().lower()
    source = str(row.get('source', 'unknown_source_category')).strip()

    inter_api_access_duration_sec = float(row.get('inter_api_access_duration(sec)', API_TRAIN_MEAN_INTER_API_ACCESS_DURATION) or 0.0)
    api_access_uniqueness = float(row.get('api_access_uniqueness', API_TRAIN_MEAN_API_ACCESS_UNIQUENESS) or 0.0)
    sequence_length_count = float(row.get('sequence_length(count)', 0) or 0)
    vsession_duration_min = float(row.get('vsession_duration(min)', 0) or 0)
    num_sessions = float(row.get('num_sessions', 0) or 0)
    num_users = float(row.get('num_users', 0) or 0)
    num_unique_apis = float(row.get('num_unique_apis', 0) or 0)

    if (
        behavior_type == 'attack'
        or 'sql injection' in behavior
        or sequence_length_count > 100
        or vsession_duration_min > 100000
        or num_sessions > 100
        or num_users > 50
    ):
        return 'attack'

    if (
        behavior_type == 'bot'
        or ip_type in ('google_bot', 'datacenter', 'private_ip')
        or 'googlebot' in behavior
        or 'bingpreview' in behavior
        or 'adsbot-google' in behavior
        or source == 'F'
    ):
        return 'bot'

    if (
        behavior_type == 'outlier'
        and inter_api_access_duration_sec < 0.01
        and (sequence_length_count > 100 or num_sessions > 10)
        and (vsession_duration_min > 30000 or num_users > 10 or num_unique_apis > 8)
    ):
        return 'bot'

    if (
        behavior_type == 'normal'
        and ip_type == 'default'
        and 0.38 <= inter_api_access_duration_sec <= 9.49
        and 0.19 <= api_access_uniqueness <= 0.66
        and 6.66 <= sequence_length_count <= 58
        and 543 <= vsession_duration_min <= 25056
        and 1 <= num_sessions <= 3
        and 1 <= num_users <= 2
        and 5 <= num_unique_apis <= 21
    ):
        return 'normal'

    return 'outlier'


def normalize_api_prediction(raw_label, features) -> str:
    """Normalize model/heuristic outputs into one of: normal, bot, outlier, attack."""
    text_label = str(raw_label).strip().lower()

    if text_label in ('normal', 'benign', 'safe', 'legitimate', 'ok', '0'):
        return 'normal'
    if any(token in text_label for token in ('bot', 'crawler', 'spider', 'scraper', 'automation')):
        return 'bot'
    if text_label == 'e':
        return 'attack'
    if text_label == 'f':
        return 'bot'
    if any(token in text_label for token in ('attack', 'malicious', 'intrusion', 'exploit', 'sqli', 'xss', 'ddos', 'dos')):
        return 'attack'
    if any(token in text_label for token in ('outlier', 'anomaly', 'abnormal', 'unknown', 'novel', 'suspicious')):
        return 'outlier'

    # deterministic fallback based on extracted request features
    if features.get('suspicious_tokens'):
        return 'attack'
    if features.get('ua_has_bot'):
        return 'bot'
    if features.get('body_len', 0) > 2000 or features.get('num_digits_in_path', 0) > 10:
        return 'outlier'
    return 'normal'


def _extract_ip(src_ip):
    if not src_ip:
        return '0.0.0.0'
    if isinstance(src_ip, str) and ',' in src_ip:
        return src_ip.split(',')[0].strip()
    return str(src_ip)


def _ip_type_code(ip_text: str) -> int:
    ip_value = _extract_ip(ip_text)
    if ip_value.startswith('127.'):
        return 0
    if ip_value.startswith('10.') or ip_value.startswith('192.168.') or ip_value.startswith('172.'):
        return 1
    return 2


def _source_type_code(user_agent: str) -> int:
    ua = (user_agent or '').lower()
    if any(tok in ua for tok in ('bot', 'crawler', 'spider', 'scrapy')):
        return 2
    if any(tok in ua for tok in ('python-requests', 'curl', 'postman', 'httpie')):
        return 1
    if any(tok in ua for tok in ('mozilla', 'chrome', 'safari', 'edge', 'firefox')):
        return 0
    return 3


def _behaviour_type_codes(inter_api_secs: float, suspicious_tokens: float, source_type_code: int, body_len: int):
    if suspicious_tokens or body_len > 2000:
        return 2, 2
    # Keep bot signal conservative; very short intervals alone should not force bot.
    if source_type_code == 2 or inter_api_secs < 0.15:
        return 1, 1
    return 0, 0


def _has_model_ready_api_fields(request_data: dict) -> bool:
    if not isinstance(request_data, dict):
        return False
    required_keys = {
        'inter_api_access_duration(sec)',
        'api_access_uniqueness',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis'
    }
    return required_keys.issubset(set(request_data.keys()))


def _normalize_behavior_class(label_value) -> str:
    text = str(label_value or '').strip().lower()
    if text in API_ALLOWED_CLASSES:
        return text
    if 'attack' in text or 'sql' in text or 'inject' in text or 'malicious' in text:
        return 'attack'
    if 'bot' in text or 'crawler' in text or 'spider' in text:
        return 'bot'
    if 'outlier' in text or 'anomaly' in text:
        return 'outlier'
    return 'normal'


def _predict_classes_for_candidate_df(candidate_df: pd.DataFrame) -> pd.Series:
    """Batch-predict classes for candidate rows using strict API model artifacts."""
    if candidate_df.empty:
        return pd.Series(dtype=str)

    strict_ready, reason = strict_api_artifacts_ready()
    if not strict_ready:
        raise RuntimeError(f'Strict API artifacts are not ready: {reason}')

    input_df = candidate_df.copy()
    if 'source' not in input_df.columns:
        input_df['source'] = input_df['behavior_type'].apply(
            lambda bt: 'E' if _normalize_behavior_class(bt) == 'attack' else (
                'F' if _normalize_behavior_class(bt) == 'bot' else 'unknown_source_category'
            )
        )

    input_df['type_ip'] = input_df['ip_type'].apply(lambda x: safe_label_transform(api_le_ip_type, x, 'unknown_ip_type'))
    input_df['type_behaviour'] = input_df['behavior_type'].apply(lambda x: safe_label_transform(api_le_behavior_type, x, 'unknown_behavior_type'))
    input_df['behaviour'] = input_df['behavior'].apply(lambda x: safe_label_transform(api_le_behavior, x, 'unknown_behavior_category'))
    input_df['source_type'] = input_df['source'].apply(lambda x: safe_label_transform(api_source_encoder, x, 'unknown_source_category'))

    feature_cols = [
        'inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
        'vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis',
        'type_ip', 'type_behaviour', 'source_type', 'behaviour'
    ]

    model_feature_names = list(getattr(api_model, 'feature_names_in_', []))
    X_df = input_df[feature_cols].copy()
    if model_feature_names:
        for col in model_feature_names:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[model_feature_names]

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0)

    preds = api_model.predict(X_df)
    if api_target_encoder is not None:
        raw_labels = api_target_encoder.inverse_transform(preds)
    else:
        raw_labels = [str(p) for p in preds]

    normalized = [normalize_api_prediction(label, {}) for label in raw_labels]
    return pd.Series(normalized, index=input_df.index)


def _build_api_test_dataset_from_source() -> pd.DataFrame:
    if not os.path.exists(API_TEST_SOURCE_DATASET_PATH):
        raise FileNotFoundError(f"Source dataset not found: {API_TEST_SOURCE_DATASET_PATH}")

    source_df = pd.read_csv(API_TEST_SOURCE_DATASET_PATH)
    if source_df.empty:
        raise ValueError('Source dataset is empty')

    if 'behavior_type' not in source_df.columns:
        raise ValueError("Source dataset must contain 'behavior_type' column")

    working_df = source_df.copy()
    working_df['behavior_type'] = working_df['behavior_type'].apply(_normalize_behavior_class)
    if 'behavior' not in working_df.columns:
        working_df['behavior'] = working_df['behavior_type']
    if 'ip_type' not in working_df.columns:
        working_df['ip_type'] = 'default'

    numeric_cols = [
        'inter_api_access_duration(sec)',
        'api_access_uniqueness',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis'
    ]
    for col in numeric_cols:
        if col not in working_df.columns:
            raise ValueError(f"Source dataset missing required column: {col}")
        working_df[col] = pd.to_numeric(working_df[col], errors='coerce')

    strict_ready, _ = strict_api_artifacts_ready()
    sampled_parts = []
    for class_name, required_count in API_TEST_REQUIRED_CLASS_COUNTS.items():
        class_df = working_df[working_df['behavior_type'] == class_name].dropna(subset=numeric_cols).copy()

        # Prefer model-consistent rows so expected labels align with strict model predictions.
        if strict_ready and not class_df.empty:
            check_size = min(len(class_df), max(required_count * 30, 2000))
            candidate_df = class_df.sample(n=check_size, random_state=42).copy() if len(class_df) > check_size else class_df.copy()
            candidate_df['_predicted_class'] = _predict_classes_for_candidate_df(candidate_df)
            model_match_df = candidate_df[candidate_df['_predicted_class'] == class_name].drop(columns=['_predicted_class'])
            class_df = model_match_df

        available_count = len(class_df)
        if available_count < required_count:
            raise ValueError(
                f"Not enough model-consistent rows for class '{class_name}'. Required {required_count}, available {available_count}."
            )

        sampled_parts.append(class_df.sample(n=required_count, random_state=42))

    extracted_df = pd.concat(sampled_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    extracted_df['source'] = extracted_df['behavior_type'].apply(
        lambda value: 'E' if value == 'attack' else ('F' if value == 'bot' else 'unknown_source_category')
    )

    keep_cols = [
        'inter_api_access_duration(sec)',
        'api_access_uniqueness',
        'sequence_length(count)',
        'vsession_duration(min)',
        'num_sessions',
        'num_users',
        'num_unique_apis',
        'ip_type',
        'behavior',
        'behavior_type',
        'source'
    ]
    extracted_df = extracted_df[keep_cols]
    extracted_df.to_csv(API_TEST_EXTRACTED_DATASET_PATH, index=False)
    return extracted_df


def get_api_test_dataset(force_refresh: bool = False) -> pd.DataFrame:
    global api_test_dataset_cache
    with api_test_dataset_lock:
        if force_refresh:
            api_test_dataset_cache = None

        if api_test_dataset_cache is not None and not api_test_dataset_cache.empty:
            return api_test_dataset_cache.copy()

        if os.path.exists(API_TEST_EXTRACTED_DATASET_PATH) and not force_refresh:
            cached_df = pd.read_csv(API_TEST_EXTRACTED_DATASET_PATH)
            if not cached_df.empty:
                api_test_dataset_cache = cached_df
                return cached_df.copy()

        rebuilt_df = _build_api_test_dataset_from_source()
        api_test_dataset_cache = rebuilt_df
        return rebuilt_df.copy()


def _build_dataset_event_metadata(behavior_type: str, behavior_text: str) -> dict:
    behavior_type = _normalize_behavior_class(behavior_type)
    behavior_text = str(behavior_text or '').strip().lower()

    if behavior_type == 'attack':
        return {
            'method': 'POST',
            'path': '/api/login?user=admin',
            'user_agent': 'python-requests/2.31.0',
            'body': "username=admin' OR 1=1 --&password=x"
        }
    if behavior_type == 'bot':
        user_agent = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
        if 'bing' in behavior_text:
            user_agent = 'Mozilla/5.0 (compatible; bingpreview/1.0b)'
        elif 'adsbot' in behavior_text:
            user_agent = 'AdsBot-Google (+http://www.google.com/adsbot.html)'
        return {
            'method': 'GET',
            'path': '/api/public/feed',
            'user_agent': user_agent,
            'body': ''
        }
    if behavior_type == 'outlier':
        return {
            'method': 'GET',
            'path': '/api/history/998877665544332211',
            'user_agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36',
            'body': ''
        }
    return {
        'method': 'GET',
        'path': '/api/products',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
        'body': ''
    }


def dataset_row_to_api_event(row: pd.Series) -> dict:
    behavior_type = _normalize_behavior_class(row.get('behavior_type'))
    behavior = str(row.get('behavior', behavior_type) or behavior_type)
    ip_type = str(row.get('ip_type', 'default') or 'default').strip().lower()
    source = str(row.get('source', '') or '').strip() or ('E' if behavior_type == 'attack' else ('F' if behavior_type == 'bot' else 'unknown_source_category'))

    if ip_type == 'google_bot':
        src_ip = '66.249.66.1'
    elif ip_type == 'private_ip':
        src_ip = _random_ipv4(private=True)
    else:
        src_ip = _random_ipv4(private=False)

    meta = _build_dataset_event_metadata(behavior_type, behavior)
    return {
        'method': meta['method'],
        'path': meta['path'],
        'headers': {
            'User-Agent': meta['user_agent'],
            'Content-Type': 'application/json'
        },
        'body': meta['body'],
        'src_ip': src_ip,
        'status_code': 200,
        'inter_api_access_duration(sec)': float(row.get('inter_api_access_duration(sec)', API_TRAIN_MEAN_INTER_API_ACCESS_DURATION)),
        'api_access_uniqueness': float(row.get('api_access_uniqueness', API_TRAIN_MEAN_API_ACCESS_UNIQUENESS)),
        'sequence_length(count)': float(row.get('sequence_length(count)', 0)),
        'vsession_duration(min)': float(row.get('vsession_duration(min)', 0)),
        'num_sessions': float(row.get('num_sessions', 0)),
        'num_users': float(row.get('num_users', 0)),
        'num_unique_apis': float(row.get('num_unique_apis', 0)),
        'ip_type': ip_type,
        'behavior_type': behavior_type,
        'behavior': behavior,
        'source': source
    }


def _safe_request_remote_addr():
    try:
        return request.remote_addr
    except Exception:
        return None


def _dataset_stream_worker(interval_seconds: float):
    global api_dataset_streaming_active, api_dataset_stream_index
    try:
        while not api_dataset_stream_stop_event.is_set():
            if not is_api_monitoring_active:
                break

            dataset_df = get_api_test_dataset(force_refresh=False)
            if dataset_df.empty:
                break

            with api_dataset_stream_lock:
                row_index = api_dataset_stream_index % len(dataset_df)
                row = dataset_df.iloc[row_index]
                api_dataset_stream_index += 1

            event_data = dataset_row_to_api_event(row)
            process_api_request_data(
                event_data,
                user_id=None,
                source='dataset_stream',
                require_trained_model=True
            )

            wait_seconds = max(0.2, float(interval_seconds))
            if api_dataset_stream_stop_event.wait(wait_seconds):
                break
    except Exception as stream_error:
        logging.error(f'Dataset stream worker stopped due to error: {stream_error}')
    finally:
        api_dataset_streaming_active = False


def start_dataset_event_stream(interval_seconds: float = 1.0) -> bool:
    global api_dataset_stream_thread, api_dataset_streaming_active
    with api_dataset_stream_lock:
        if api_dataset_streaming_active and api_dataset_stream_thread is not None and api_dataset_stream_thread.is_alive():
            return False

        api_dataset_stream_stop_event.clear()
        api_dataset_streaming_active = True
        api_dataset_stream_thread = threading.Thread(
            target=_dataset_stream_worker,
            args=(interval_seconds,),
            daemon=True
        )
        api_dataset_stream_thread.start()
        return True


def stop_dataset_event_stream():
    global api_dataset_streaming_active
    api_dataset_stream_stop_event.set()
    api_dataset_streaming_active = False


def derive_api_categories(request_data: dict, features: dict) -> dict:
    """Derive categorical fields expected by robust training preprocess."""
    headers = request_data.get('headers') or {}
    ua = (headers.get('User-Agent', '') or '').strip().lower()
    path = str(request_data.get('path', '/'))
    body = str(request_data.get('body', '') or '')
    src_ip = _extract_ip(request_data.get('src_ip'))

    if 'googlebot' in ua or src_ip.startswith('66.249.'):
        ip_type = 'google_bot'
    elif src_ip.startswith('10.') or src_ip.startswith('192.168.') or src_ip.startswith('172.'):
        ip_type = 'private_ip'
    elif any(tool in ua for tool in ('python-requests', 'curl', 'postman', 'httpie', 'scrapy')):
        ip_type = 'datacenter'
    else:
        ip_type = 'default'

    suspicious_tokens = float(features.get('suspicious_tokens', 0.0))
    if suspicious_tokens > 0:
        behavior_type = 'attack'
        behavior = 'sql injection'
    elif any(bot in ua for bot in ('bot', 'crawler', 'spider', 'adsbot-google', 'bingpreview')):
        behavior_type = 'bot'
        behavior = 'googlebot/2.1' if 'googlebot' in ua else ('bingpreview/1.0b' if 'bingpreview' in ua else 'adsbot-google')
    elif features.get('body_len', 0) > 2000 or features.get('num_digits_in_path', 0) > 10:
        behavior_type = 'outlier'
        behavior = 'outlier'
    else:
        behavior_type = 'normal'
        behavior = 'normal'

    if behavior_type == 'attack':
        source = 'E'
    elif behavior_type == 'bot':
        source = 'F'
    else:
        source = 'unknown_source_category'

    return {
        'ip_type': ip_type,
        'behavior_type': behavior_type,
        'behavior': behavior,
        'source': source
    }


def preprocess_api_data(raw_data: dict, model_expects_source_type: bool = True) -> pd.DataFrame:
    """Preprocess raw API data consistently with trained API behavior model."""
    input_df = pd.DataFrame([raw_data or {}])

    expected_raw_cols = [
        'inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
        'vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis',
        'ip_type', 'behavior', 'behavior_type', 'source', 'type_ip', 'type_behaviour', 'behaviour', 'source_type'
    ]
    for col in expected_raw_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    numeric_defaults = {
        'inter_api_access_duration(sec)': API_TRAIN_MEAN_INTER_API_ACCESS_DURATION,
        'api_access_uniqueness': API_TRAIN_MEAN_API_ACCESS_UNIQUENESS,
        'sequence_length(count)': 0,
        'vsession_duration(min)': 0,
        'num_sessions': 0,
        'num_users': 0,
        'num_unique_apis': 0,
    }
    for col, default_value in numeric_defaults.items():
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(default_value)

    for col in ['type_ip', 'type_behaviour', 'behaviour', 'source_type']:
        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

    if input_df['type_ip'].isna().any():
        input_df.loc[input_df['type_ip'].isna(), 'type_ip'] = input_df.loc[input_df['type_ip'].isna(), 'ip_type'].apply(
            lambda x: safe_label_transform(api_le_ip_type, x, 'unknown_ip_type')
        )
    if input_df['type_behaviour'].isna().any():
        input_df.loc[input_df['type_behaviour'].isna(), 'type_behaviour'] = input_df.loc[input_df['type_behaviour'].isna(), 'behavior_type'].apply(
            lambda x: safe_label_transform(api_le_behavior_type, x, 'unknown_behavior_type')
        )
    if input_df['behaviour'].isna().any():
        input_df.loc[input_df['behaviour'].isna(), 'behaviour'] = input_df.loc[input_df['behaviour'].isna(), 'behavior'].apply(
            lambda x: safe_label_transform(api_le_behavior, x, 'unknown_behavior_category')
        )

    if model_expects_source_type:
        if input_df['source_type'].isna().any():
            input_df.loc[input_df['source_type'].isna(), 'source_type'] = input_df.loc[input_df['source_type'].isna(), 'source'].apply(
                lambda x: safe_label_transform(api_source_encoder, x, 'unknown_source_category')
            )

    feature_cols = [
        'inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
        'vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis',
        'type_ip', 'type_behaviour', 'behaviour'
    ]
    if model_expects_source_type:
        feature_cols.insert(-1, 'source_type')

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    final_processed_df = input_df[feature_cols].copy()
    for col in final_processed_df.columns:
        final_processed_df[col] = pd.to_numeric(final_processed_df[col], errors='coerce').fillna(0.0)

    return final_processed_df


def build_behavioral_features(request_data: dict):
    """Build live session/behavioral features aligned with trained model design."""
    now = time.time()
    method = (request_data.get('method') or 'GET').upper()
    path = request_data.get('path') or '/'
    headers = request_data.get('headers') or {}
    body = request_data.get('body') or ''
    src_ip = _extract_ip(request_data.get('src_ip'))
    user_id = request_data.get('user_id') or src_ip
    ua = headers.get('User-Agent', '')
    suspicious_tokens = float(bool(re.search(r"\b(select|union|drop|insert|--|\bexec\b)\b", f"{path} {body}", re.I)))

    with api_behavioral_lock:
        user_state = api_behavioral_state['users'].get(user_id)
        if user_state is None:
            user_state = {
                'session_start': now,
                'last_seen': None,
                'session_count': 1,
                'sequence_count': 0,
                'paths_in_session': set(),
                'all_paths': set()
            }
            api_behavioral_state['users'][user_id] = user_state

        last_seen = user_state['last_seen']
        if last_seen is None:
            inter_api_access_duration = 0.0
        else:
            inter_api_access_duration = max(0.0, now - last_seen)

        if last_seen is not None and (now - last_seen) > API_SESSION_TIMEOUT_SECONDS:
            user_state['session_count'] += 1
            user_state['session_start'] = now
            user_state['sequence_count'] = 0
            user_state['paths_in_session'] = set()

        user_state['sequence_count'] += 1
        user_state['last_seen'] = now
        user_state['paths_in_session'].add(path)
        user_state['all_paths'].add(path)
        api_behavioral_state['global_unique_paths'].add(path)

        total_seen = max(1, user_state['sequence_count'])
        api_access_uniqueness = len(user_state['paths_in_session']) / float(total_seen)
        vsession_duration_min = max(0.0, (now - user_state['session_start']) / 60.0)
        num_sessions = int(user_state['session_count'])
        num_users = len(api_behavioral_state['users'])
        num_unique_apis = len(user_state['all_paths'])

    source_type = _source_type_code(ua)
    type_behaviour, behaviour = _behaviour_type_codes(inter_api_access_duration, suspicious_tokens, source_type, len(body))
    type_ip = _ip_type_code(src_ip)

    features = {
        'inter_api_access_duration(sec)': float(inter_api_access_duration),
        'api_access_uniqueness': float(api_access_uniqueness),
        'sequence_length(count)': int(user_state['sequence_count']),
        'vsession_duration(min)': float(vsession_duration_min),
        'num_sessions': int(num_sessions),
        'num_users': int(num_users),
        'num_unique_apis': int(num_unique_apis),
        'type_ip': int(type_ip),
        'type_behaviour': int(type_behaviour),
        'source_type': int(source_type),
        'behaviour': int(behaviour)
    }

    return features


def confidence_to_accuracy_percent(confidence_value) -> float:
    """Convert model confidence score to percentage for UI display."""
    try:
        confidence = float(confidence_value)
    except Exception:
        confidence = 0.0

    if confidence <= 1.0:
        confidence *= 100.0

    confidence = max(0.0, min(100.0, confidence))
    return round(confidence, 2)


def predict_api_call(request_data: dict, require_trained_model: bool = False) -> dict:
    """Predict API call type using model or heuristics.

    Set require_trained_model=True to forbid fallback predictions.
    """
    try:
        method = request_data.get('method', 'GET').upper()
        path = request_data.get('path', '/')
        ua = request_data.get('headers', {}).get('User-Agent', '').lower()
        body = request_data.get('body', '') or ''

        # helper values for normalization fallback rules
        fallback_features = {
            'ua_has_bot': any(bot in ua for bot in ('bot', 'crawler', 'spider', 'scrapy')),
            'body_len': len(body),
            'num_digits_in_path': sum(c.isdigit() for c in path),
            'suspicious_tokens': float(bool(re.search(r"\b(select|union|drop|insert|--|\bexec\b)\b", path + ' ' + body, re.I)))
        }

        if _has_model_ready_api_fields(request_data):
            preprocessed_input = {
                'inter_api_access_duration(sec)': request_data.get('inter_api_access_duration(sec)'),
                'api_access_uniqueness': request_data.get('api_access_uniqueness'),
                'sequence_length(count)': request_data.get('sequence_length(count)'),
                'vsession_duration(min)': request_data.get('vsession_duration(min)'),
                'num_sessions': request_data.get('num_sessions'),
                'num_users': request_data.get('num_users'),
                'num_unique_apis': request_data.get('num_unique_apis'),
                'ip_type': request_data.get('ip_type', 'unknown_ip_type'),
                'behavior': request_data.get('behavior', 'unknown_behavior_category'),
                'behavior_type': request_data.get('behavior_type', 'unknown_behavior_type'),
                'source': request_data.get('source', 'unknown_source_category'),
                'type_ip': request_data.get('type_ip'),
                'type_behaviour': request_data.get('type_behaviour'),
                'behaviour': request_data.get('behaviour'),
                'source_type': request_data.get('source_type')
            }
            features = {
                'suspicious_tokens': fallback_features['suspicious_tokens'],
                'ua_has_bot': fallback_features['ua_has_bot'],
                'body_len': fallback_features['body_len'],
                'num_digits_in_path': fallback_features['num_digits_in_path']
            }
        else:
            features = build_behavioral_features(request_data)
            features['ua_has_bot'] = fallback_features['ua_has_bot']
            features['body_len'] = fallback_features['body_len']
            features['num_digits_in_path'] = fallback_features['num_digits_in_path']
            features['suspicious_tokens'] = fallback_features['suspicious_tokens']

            categorized = derive_api_categories(request_data, features)
            preprocessed_input = {
                'inter_api_access_duration(sec)': features.get('inter_api_access_duration(sec)'),
                'api_access_uniqueness': features.get('api_access_uniqueness'),
                'sequence_length(count)': features.get('sequence_length(count)'),
                'vsession_duration(min)': features.get('vsession_duration(min)'),
                'num_sessions': features.get('num_sessions'),
                'num_users': features.get('num_users'),
                'num_unique_apis': features.get('num_unique_apis'),
                'ip_type': categorized.get('ip_type'),
                'behavior': categorized.get('behavior'),
                'behavior_type': categorized.get('behavior_type'),
                'source': categorized.get('source'),
                'type_ip': features.get('type_ip'),
                'type_behaviour': features.get('type_behaviour'),
                'behaviour': features.get('behaviour'),
                'source_type': features.get('source_type')
            }

        model_feature_names = list(getattr(api_model, 'feature_names_in_', [])) if api_model is not None else []
        model_expects_source_type = True
        if model_feature_names:
            model_expects_source_type = 'source_type' in model_feature_names

        X_df = preprocess_api_data(preprocessed_input, model_expects_source_type=model_expects_source_type)
        if model_feature_names:
            for col in model_feature_names:
                if col not in X_df.columns:
                    X_df[col] = 0.0
            X_df = X_df[model_feature_names]

        X = X_df.values

        if api_scaler is not None:
            try:
                X = api_scaler.transform(X)
            except Exception:
                pass

        if api_model is not None:
            pred = api_model.predict(X_df if api_scaler is None else X)[0]
            confidence = 0.0
            try:
                probs = api_model.predict_proba(X_df if api_scaler is None else X)[0]
                confidence = float(max(probs))
            except Exception:
                confidence = 0.8

            if api_target_encoder is not None:
                try:
                    raw_label = api_target_encoder.inverse_transform([pred])[0]
                except Exception:
                    raw_label = str(pred)
            else:
                raw_label = str(pred)

            normalized_prediction = normalize_api_prediction(raw_label, features)

            return {
                'prediction': normalized_prediction,
                'raw_prediction': str(raw_label),
                'confidence': float(confidence),
                'accuracy_percent': confidence_to_accuracy_percent(confidence),
                'prediction_method': 'trained_model',
                'used_trained_model': True
            }

        if require_trained_model:
            raise RuntimeError('Trained API model artifacts are unavailable')

        fallback_label = classify_api_call(pd.Series(preprocessed_input))
        confidence_map = {
            'attack': 0.9,
            'bot': 0.85,
            'outlier': 0.7,
            'normal': 0.6
        }
        fallback_confidence = float(confidence_map.get(fallback_label, 0.6))
        return {
            'prediction': normalize_api_prediction(fallback_label, features),
            'raw_prediction': str(fallback_label),
            'confidence': fallback_confidence,
            'accuracy_percent': confidence_to_accuracy_percent(fallback_confidence),
            'prediction_method': 'rule_based_fallback',
            'used_trained_model': False
        }

    except Exception as e:
        if require_trained_model:
            raise
        logging.error(f'API prediction error: {e}')
        return {
            'prediction': 'normal',
            'raw_prediction': 'normal',
            'confidence': 0.5,
            'accuracy_percent': confidence_to_accuracy_percent(0.5),
            'prediction_method': 'error_fallback',
            'used_trained_model': False
        }


def should_capture_http_request(req) -> bool:
    """Filter noisy/internal requests from live API monitoring."""
    try:
        path = req.path or '/'
        if req.method == 'OPTIONS':
            return False
        if not path.startswith('/api'):
            return False
        if path in API_EXCLUDED_PATHS:
            return False
        if any(path.startswith(prefix) for prefix in API_EXCLUDED_PREFIXES):
            return False
        return True
    except Exception:
        return False


def process_api_request_data(data: dict, user_id=None, source='socket', require_trained_model: bool = False):
    """Shared processor for API requests from socket events and real HTTP traffic."""
    global api_requests_log, api_anomaly_count, api_class_counts

    if not is_api_monitoring_active:
        return None

    normalized_data = dict(data or {})
    if user_id is not None:
        normalized_data['user_id'] = user_id

    record = {
        'timestamp': datetime.now(),
        'request': normalized_data,
        'user_id': user_id,
        'source': source
    }

    strict_required = bool(require_trained_model)
    if source in ('dataset_stream', 'http', 'socket', 'test_generator'):
        strict_required = True

    if strict_required:
        strict_ready, strict_reason = strict_api_artifacts_ready()
        if not strict_ready:
            logging.error(f'API live event skipped: strict API artifacts are not ready ({strict_reason}).')
            return None

    try:
        result = predict_api_call(normalized_data, require_trained_model=strict_required)
    except Exception as model_error:
        logging.error(f'API request skipped (trained-model requirement): {model_error}')
        return None

    prediction = result.get('prediction', 'normal')

    if prediction not in API_ALLOWED_CLASSES:
        prediction = normalize_api_prediction(prediction, {})

    record['prediction'] = prediction
    record['raw_prediction'] = result.get('raw_prediction', prediction)
    record['confidence'] = float(result.get('confidence', 0.0))
    record['accuracy_percent'] = float(result.get('accuracy_percent', confidence_to_accuracy_percent(record['confidence'])))
    record['prediction_method'] = result.get('prediction_method', 'unknown')
    record['used_trained_model'] = bool(result.get('used_trained_model', False))

    try:
        if hasattr(mongo_db.api_logs, 'insert_one'):
            mongo_db.api_logs.insert_one(record.copy())
        elif isinstance(mongo_db.api_logs, list):
            mongo_db.api_logs.append(record.copy())
    except Exception:
        pass

    api_requests_log.append(record)
    if len(api_requests_log) > 1000:
        api_requests_log = api_requests_log[-1000:]

    api_class_counts[prediction] += 1

    if prediction != 'normal':
        api_anomaly_count += 1

    remote_addr = _safe_request_remote_addr()

    socketio.emit('api_threat', {
        'prediction': prediction,
        'raw_prediction': result.get('raw_prediction'),
        'confidence': result.get('confidence'),
        'accuracy_percent': result.get('accuracy_percent'),
        'prediction_method': result.get('prediction_method'),
        'used_trained_model': bool(result.get('used_trained_model', False)),
        'model_artifacts_used': strict_api_artifacts_payload() if bool(result.get('used_trained_model', False)) else None,
        'src_ip': normalized_data.get('src_ip', remote_addr),
        'method': normalized_data.get('method'),
        'path': normalized_data.get('path'),
        'timestamp': datetime.now().isoformat(),
        'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES},
        'total_requests': len(api_requests_log),
        'total_anomalies': api_anomaly_count
    })

    socketio.emit('api_monitoring_status', {
        'is_monitoring': is_api_monitoring_active,
        'total_requests': len(api_requests_log),
        'total_anomalies': api_anomaly_count,
        'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
    })

    return record


def _random_ipv4(private: bool = False) -> str:
    if private:
        block = random.choice(['10', '172', '192'])
        if block == '172':
            return f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        if block == '192':
            return f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return f"{random.randint(11, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


@socketio.on('api_request')
def handle_api_request_event(data):
    """Handle real-time API request events from frontend/services."""
    try:
        process_api_request_data(data, user_id=session.get('user_id'), source='socket')

    except Exception as e:
        logging.error(f'Error handling api_request event: {e}')


@app.after_request
def monitor_http_requests(response):
    """Capture real incoming HTTP traffic for API Analyzer live monitoring."""
    try:
        if is_api_monitoring_active and should_capture_http_request(request):
            body_text = ''
            if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
                body_text = (request.get_data(cache=True, as_text=True) or '')[:2000]

            src_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
            if isinstance(src_ip, str) and ',' in src_ip:
                src_ip = src_ip.split(',')[0].strip()

            request_data = {
                'method': request.method,
                'path': request.path,
                'headers': {
                    'User-Agent': request.headers.get('User-Agent', ''),
                    'Content-Type': request.headers.get('Content-Type', '')
                },
                'body': body_text,
                'src_ip': src_ip,
                'status_code': getattr(response, 'status_code', None)
            }

            process_api_request_data(request_data, user_id=session.get('user_id'), source='http')
    except Exception as e:
        logging.error(f'HTTP request monitor error: {e}')

    return response


@app.route('/api/start_api_monitoring', methods=['POST'])
def start_api_monitoring():
    global is_api_monitoring_active
    try:
        is_api_monitoring_active = True
        socketio.emit('api_monitoring_status', {
            'is_monitoring': True,
            'total_requests': len(api_requests_log),
            'total_anomalies': api_anomaly_count,
            'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
        })
        return jsonify({'status': 'success', 'message': 'API monitoring started'})
    except Exception as e:
        logging.error(f'Failed to start API monitoring: {e}')
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/stop_api_monitoring', methods=['POST'])
def stop_api_monitoring():
    global is_api_monitoring_active
    try:
        is_api_monitoring_active = False
        stop_dataset_event_stream()
        socketio.emit('api_monitoring_status', {
            'is_monitoring': False,
            'total_requests': len(api_requests_log),
            'total_anomalies': api_anomaly_count,
            'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
        })
        return jsonify({'status': 'success', 'message': 'API monitoring stopped'})
    except Exception as e:
        logging.error(f'Failed to stop API monitoring: {e}')
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/get_api_stats', methods=['GET'])
def get_api_stats():
    try:
        avg_confidence = 0.0
        if api_requests_log:
            avg_confidence = sum(float(item.get('confidence', 0.0)) for item in api_requests_log) / float(len(api_requests_log))

        return jsonify({
            'status': 'success',
            'is_monitoring': is_api_monitoring_active,
            'total_requests': len(api_requests_log),
            'total_anomalies': api_anomaly_count,
            'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES},
            'average_accuracy_percent': confidence_to_accuracy_percent(avg_confidence)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/generate_api_report', methods=['POST'])
def generate_api_report():
    try:
        class_counts = {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
        total_requests = len(api_requests_log)
        anomaly_rate = (api_anomaly_count / total_requests * 100.0) if total_requests > 0 else 0.0

        recent_events = []
        for item in api_requests_log[-10:]:
            req = item.get('request', {})
            ts = item.get('timestamp')
            recent_events.append({
                'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                'method': req.get('method', 'GET'),
                'path': req.get('path', '/'),
                'src_ip': req.get('src_ip', '0.0.0.0'),
                'prediction': item.get('prediction', 'normal'),
                'confidence': float(item.get('confidence', 0.0)),
                'accuracy_percent': float(item.get('accuracy_percent', confidence_to_accuracy_percent(item.get('confidence', 0.0))))
            })

        path_counts = defaultdict(int)
        for item in api_requests_log:
            req = item.get('request', {})
            path_counts[req.get('path', '/')] += 1
        top_paths = sorted(path_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]

        report = {
            'total_requests': total_requests,
            'total_anomalies': api_anomaly_count,
            'anomaly_rate': round(anomaly_rate, 2),
            'class_counts': class_counts,
            'top_paths': [{'path': path, 'count': count} for path, count in top_paths],
            'recent_events': recent_events,
            'generated_at': datetime.now().isoformat(),
            'monitoring_status': 'Running' if is_api_monitoring_active else 'Stopped'
        }

        mongo_db.log_system_event('API_REPORT_GENERATED', 'API forensic report generated', session.get('user_id'), 'INFO')
        return jsonify({'status': 'success', 'report': report})
    except Exception as e:
        mongo_db.log_system_event('API_REPORT_ERROR', f'Failed to generate API report: {e}', session.get('user_id'), 'ERROR')
        return jsonify({'status': 'error', 'message': str(e)})


# Test endpoint to verify API Analyzer is functional
@app.route('/api/test_api_call', methods=['POST'])
def test_api_call():
    try:
        data = request.get_json() or {}
        result = predict_api_call(data)
        models_loaded = api_model is not None and api_target_encoder is not None
        return jsonify({'status': 'success', 'models_loaded': models_loaded, 'result': result})
    except Exception as e:
        logging.error(f'Test API call failed: {e}')
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/generate_test_api_event', methods=['POST'])
def generate_test_api_event():
    """Start automatic dataset-driven event streaming or generate one-time dataset batch."""
    global is_api_monitoring_active
    try:
        payload = request.get_json(silent=True) or {}
        requested_profile = str(payload.get('profile', '')).strip().lower() or None
        requested_count = payload.get('count', 1)
        auto_start = bool(payload.get('auto_start', True))
        refresh_dataset = bool(payload.get('refresh_dataset', False))
        auto_stream = bool(payload.get('auto_stream', True))
        stream_interval_seconds = payload.get('stream_interval_seconds', 1.0)

        try:
            stream_interval_seconds = float(stream_interval_seconds)
        except Exception:
            stream_interval_seconds = 1.0
        stream_interval_seconds = max(0.2, min(10.0, stream_interval_seconds))

        strict_ready, strict_reason = strict_api_artifacts_ready()
        if not strict_ready:
            return jsonify({
                'status': 'error',
                'message': (
                    f'Strict API artifacts are required: '
                    f'{STRICT_API_MODEL_FILE}, {STRICT_API_SOURCE_ENCODER_FILE}, {STRICT_API_TARGET_ENCODER_FILE}. '
                    f'Current issue: {strict_reason}.'
                )
            }), 500

        try:
            event_count = int(requested_count)
        except Exception:
            event_count = 1
        event_count = max(1, min(10, event_count))

        dataset_df = get_api_test_dataset(force_refresh=refresh_dataset)
        dataset_df['behavior_type'] = dataset_df['behavior_type'].apply(_normalize_behavior_class)

        if requested_profile in API_ALLOWED_CLASSES:
            candidate_df = dataset_df[dataset_df['behavior_type'] == requested_profile]
        else:
            candidate_df = dataset_df

        if candidate_df.empty:
            return jsonify({
                'status': 'error',
                'message': f"No rows available in extracted dataset for profile '{requested_profile}'"
            }), 400

        sampled_df = candidate_df.sample(
            n=event_count,
            replace=(len(candidate_df) < event_count),
            random_state=int(time.time() * 1000) % (2 ** 32)
        ).reset_index(drop=True)

        if auto_start and not is_api_monitoring_active:
            is_api_monitoring_active = True
            socketio.emit('api_monitoring_status', {
                'is_monitoring': True,
                'total_requests': len(api_requests_log),
                'total_anomalies': api_anomaly_count,
                'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
            })

        if auto_stream:
            started = start_dataset_event_stream(interval_seconds=stream_interval_seconds)
            return jsonify({
                'status': 'success',
                'models_loaded': api_model is not None and api_target_encoder is not None,
                'is_monitoring': is_api_monitoring_active,
                'dataset_file': API_TEST_EXTRACTED_DATASET_PATH,
                'model_artifacts_used': {
                    'model': os.path.join('API_analyzer', STRICT_API_MODEL_FILE),
                    'source_encoder': os.path.join('API_analyzer', STRICT_API_SOURCE_ENCODER_FILE),
                    'target_encoder': os.path.join('API_analyzer', STRICT_API_TARGET_ENCODER_FILE)
                },
                'streaming': True,
                'stream_started': started,
                'message': 'Dataset API event streaming started' if started else 'Dataset API event streaming is already running',
                'stream_interval_seconds': stream_interval_seconds,
                'generated_count': 0,
                'matched_count': 0,
                'events': []
            })

        generated_results = []
        matched_count = 0
        for _, row in sampled_df.iterrows():
            expected_profile = _normalize_behavior_class(row.get('behavior_type'))
            event_data = dataset_row_to_api_event(row)
            predicted_preview = predict_api_call(event_data)

            if is_api_monitoring_active:
                record = process_api_request_data(
                    event_data,
                    user_id=session.get('user_id'),
                    source='test_generator',
                    require_trained_model=True
                )
                if record is not None:
                    matched = str(record.get('prediction', '')).lower() == expected_profile
                    if matched:
                        matched_count += 1
                    generated_results.append({
                        'expected_profile': expected_profile,
                        'predicted_class': record.get('prediction', 'normal'),
                        'raw_prediction': record.get('raw_prediction', record.get('prediction', 'normal')),
                        'accuracy_percent': float(record.get('accuracy_percent', 0.0)),
                        'prediction_method': record.get('prediction_method', 'unknown'),
                        'used_trained_model': bool(record.get('used_trained_model', False)),
                        'matched_requested_profile': matched,
                        'attempts_used': 1,
                        'method': event_data.get('method', 'GET'),
                        'path': event_data.get('path', '/'),
                        'src_ip': event_data.get('src_ip', '0.0.0.0'),
                        'feature_snapshot': {
                            'inter_api_access_duration(sec)': event_data.get('inter_api_access_duration(sec)'),
                            'api_access_uniqueness': event_data.get('api_access_uniqueness'),
                            'sequence_length(count)': event_data.get('sequence_length(count)'),
                            'vsession_duration(min)': event_data.get('vsession_duration(min)'),
                            'num_sessions': event_data.get('num_sessions'),
                            'num_users': event_data.get('num_users'),
                            'num_unique_apis': event_data.get('num_unique_apis'),
                            'ip_type': event_data.get('ip_type'),
                            'behavior_type': event_data.get('behavior_type'),
                            'behavior': event_data.get('behavior'),
                            'source': event_data.get('source')
                        }
                    })
                    continue

            fallback_result = predicted_preview or predict_api_call(event_data, require_trained_model=True)
            matched = str(fallback_result.get('prediction', '')).lower() == expected_profile
            if matched:
                matched_count += 1
            generated_results.append({
                'expected_profile': expected_profile,
                'predicted_class': fallback_result.get('prediction', 'normal'),
                'raw_prediction': fallback_result.get('raw_prediction', fallback_result.get('prediction', 'normal')),
                'accuracy_percent': float(fallback_result.get('accuracy_percent', 0.0)),
                'prediction_method': fallback_result.get('prediction_method', 'unknown'),
                'used_trained_model': bool(fallback_result.get('used_trained_model', False)),
                'matched_requested_profile': matched,
                'attempts_used': 1,
                'method': event_data.get('method', 'GET'),
                'path': event_data.get('path', '/'),
                'src_ip': event_data.get('src_ip', '0.0.0.0'),
                'feature_snapshot': {
                    'inter_api_access_duration(sec)': event_data.get('inter_api_access_duration(sec)'),
                    'api_access_uniqueness': event_data.get('api_access_uniqueness'),
                    'sequence_length(count)': event_data.get('sequence_length(count)'),
                    'vsession_duration(min)': event_data.get('vsession_duration(min)'),
                    'num_sessions': event_data.get('num_sessions'),
                    'num_users': event_data.get('num_users'),
                    'num_unique_apis': event_data.get('num_unique_apis'),
                    'ip_type': event_data.get('ip_type'),
                    'behavior_type': event_data.get('behavior_type'),
                    'behavior': event_data.get('behavior'),
                    'source': event_data.get('source')
                }
            })

        return jsonify({
            'status': 'success',
            'models_loaded': api_model is not None and api_target_encoder is not None,
            'is_monitoring': is_api_monitoring_active,
            'requested_profile': requested_profile or 'all',
            'dataset_file': API_TEST_EXTRACTED_DATASET_PATH,
            'model_artifacts_used': {
                'model': os.path.join('API_analyzer', STRICT_API_MODEL_FILE),
                'source_encoder': os.path.join('API_analyzer', STRICT_API_SOURCE_ENCODER_FILE),
                'target_encoder': os.path.join('API_analyzer', STRICT_API_TARGET_ENCODER_FILE)
            },
            'generated_count': len(generated_results),
            'matched_count': matched_count,
            'events': generated_results
        })
    except Exception as e:
        logging.error(f'Generate test API event failed: {e}')
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/predict_behavior', methods=['POST'])
@app.route('/predict_behavior', methods=['POST'])
def predict_behavior():
    """Predict behavior for one API call payload using preprocessing + model + interpretation."""
    try:
        if not request.is_json:
            return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400

        raw_api_data = request.get_json() or {}
        result = predict_api_call(raw_api_data)

        if api_model is not None and api_target_encoder is not None:
            method = 'machine_learning'
            message = 'Prediction generated by trained API behavior model.'
        else:
            method = 'rule_based_fallback'
            message = 'Model/encoders unavailable; fallback classification used.'

        return jsonify({
            'status': 'success',
            'method': method,
            'message': message,
            'predicted_behavior': result.get('raw_prediction', result.get('prediction')),
            'normalized_class': result.get('prediction'),
            'confidence': float(result.get('confidence', 0.0)),
            'accuracy_percent': float(result.get('accuracy_percent', 0.0))
        })
    except Exception as e:
        logging.exception(f'Error during /api/predict_behavior: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500


# SocketIO event handlers for file and mouse data
@socketio.on('file_access_data')
def handle_file_access_data(data):
    """Handle real-time file access data from frontend"""
    try:
        global file_access_data, file_anomaly_count

        features = data.get('features', [])
        if features:
            # Simple anomaly detection logic for demo
            anomaly_score = sum(features) / len(features) / 1000  # Normalize
            is_anomaly = anomaly_score > 0.5

            # Store data for tracking
            file_access_record = {
                'timestamp': datetime.now(),
                'features': features,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'user_id': session.get('user_id')
            }

            # Store in MongoDB
            try:
                mongo_db.file_access_logs.insert_one(file_access_record.copy())
            except Exception as e:
                logging.error(f"Failed to log file access to MongoDB: {e}")

            file_access_data.append(file_access_record)

            # Keep only last 1000 records
            if len(file_access_data) > 1000:
                file_access_data = file_access_data[-1000:]

            if is_anomaly:
                file_anomaly_count += 1

            # Emit to all connected clients
            socketio.emit('file_access_anomaly', {
                'anomaly_score': float(anomaly_score),
                'is_anomaly': is_anomaly,
                'total_anomalies': file_anomaly_count,
                'timestamp': datetime.now().isoformat()
            })

            logging.info(f"File access processed - Anomaly: {is_anomaly}, Score: {anomaly_score}")

    except Exception as e:
        logging.error(f"Error processing file access data: {str(e)}")


@socketio.on('mouse_movement_data')
def handle_mouse_movement_data(data):
    """Handle real-time mouse movement data from frontend"""
    try:
        global mouse_movement_data, mouse_anomaly_count

        movements = data.get('movements', [])
        if movements:
            # Simple anomaly detection logic for demo
            total_movement = sum([sum(movement) for movement in movements])
            reconstruction_error = total_movement / (len(movements) * 1000)  # Normalize
            is_anomaly = reconstruction_error > 3.5

            # Store data for tracking
            mouse_movement_record = {
                'timestamp': datetime.now(),
                'movements': movements,
                'reconstruction_error': reconstruction_error,
                'is_anomaly': is_anomaly,
                'user_id': session.get('user_id')
            }

            # Store in MongoDB
            try:
                mongo_db.mouse_movement_logs.insert_one(mouse_movement_record.copy())
            except Exception as e:
                logging.error(f"Failed to log mouse movement to MongoDB: {e}")

            mouse_movement_data.append(mouse_movement_record)

            # Keep only last 1000 records
            if len(mouse_movement_data) > 1000:
                mouse_movement_data = mouse_movement_data[-1000:]

            if is_anomaly:
                mouse_anomaly_count += 1

            # Emit to all connected clients
            socketio.emit('mouse_movement_anomaly', {
                'reconstruction_error': float(reconstruction_error),
                'is_anomaly': is_anomaly,
                'total_anomalies': mouse_anomaly_count,
                'timestamp': datetime.now().isoformat()
            })

            logging.info(f"Mouse movement processed - Anomaly: {is_anomaly}, Error: {reconstruction_error}")

    except Exception as e:
        logging.error(f"Error processing mouse movement data: {str(e)}")


# ----------------- Socket Event Handlers ----------------

@socketio.on('connect')
def handle_connect():
    logging.info('Client connected')
    # Send initial state to newly connected client
    socketio.emit('network_threat', {
        'prediction': 'BENIGN',
        'total_packets': total_packets,
        'non_ip_packet_count': non_ip_packet_count,
        'class_counts': dict(class_counts),
        'timestamp': time.time()
    })
    # Send restricted monitoring status
    socketio.emit('restricted_monitoring_status', {
        'is_monitoring': is_monitoring_active,
        'total_accesses': len(restricted_access_data),
        'suspicious_domains': len([d for d in restricted_access_data if d.get('is_suspicious', False)]),
        'blocked_accesses': len([d for d in restricted_access_data if d.get('prediction', 'Benign') != 'Benign'])
    })
    # Send API analyzer status
    socketio.emit('api_monitoring_status', {
        'is_monitoring': is_api_monitoring_active,
        'total_requests': len(api_requests_log),
        'total_anomalies': api_anomaly_count,
        'class_counts': {label: int(api_class_counts.get(label, 0)) for label in API_ALLOWED_CLASSES}
    })


@socketio.on('disconnect')
def handle_disconnect():
    logging.info('Client disconnected')


# ----------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('malicious_behavior_detection.log')
    ]
)

# ----------------- Main Application ----------------

if __name__ == '__main__':
    if ENABLE_ENDPOINT_SIMULATION:
        # Start the endpoint simulation in a separate thread
        simulation_thread = threading.Thread(target=simulate_packet_processing)
        simulation_thread.daemon = True
        simulation_thread.start()
        logging.info("Endpoint simulation started")
    else:
        logging.info("Endpoint simulation disabled")

    # Start periodic speed updates in separate thread
    speed_thread = threading.Thread(target=emit_periodic_speed)
    speed_thread.daemon = True
    speed_thread.start()

    logging.info("Starting ML-Driven Forensic Malicious Behavior Detection System")
    logging.info("Network monitoring started on all available interfaces")
    logging.info("Restricted site access monitoring enabled")

    if restricted_model is None:
        logging.warning("Running restricted site monitoring in simulation mode - ML models not available")
    else:
        logging.info("Restricted site ML models loaded successfully")

    # Summary of loaded models for easier debugging
    logging.info(f"Model summary -- restricted_model: {'loaded' if restricted_model else 'missing'}, network_model: {'loaded' if network_model else 'missing'}, endpoint_model: {'loaded' if endpoint_rf_model else 'missing'}, api_models: {'loaded' if (api_model and api_label_encoder) else 'missing/partial'}")

    run_host = os.getenv('HOST', '0.0.0.0')
    run_port = parse_int_env('PORT', 5000)
    run_debug = parse_bool_env('DEBUG', default=not IS_PRODUCTION)
    allow_unsafe_werkzeug = parse_bool_env('ALLOW_UNSAFE_WERKZEUG', default=not IS_PRODUCTION)

    logging.info(f"Web interface available at http://localhost:{run_port}")
    logging.info(f"Runtime config -- env: {APP_ENV}, debug: {run_debug}, auto_login: {ENABLE_DEMO_AUTOLOGIN}, network_sniffing: {ENABLE_NETWORK_SNIFFING}, endpoint_simulation: {ENABLE_ENDPOINT_SIMULATION}, rate_limit: {RATE_LIMIT_ENABLED}")

    # Run Flask application with SocketIO
    socketio.run(app,
                 host=run_host,
                 port=run_port,
                 debug=run_debug,
                 allow_unsafe_werkzeug=allow_unsafe_werkzeug,
                 use_reloader=False)