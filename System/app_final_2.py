import logging
import os
import secrets
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
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def parse_int_env(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*" if not IS_PRODUCTION else "http://localhost:5000")
MAX_CONTENT_LENGTH = parse_int_env("MAX_CONTENT_LENGTH", 1024 * 1024)
ENABLE_DEMO_AUTOLOGIN = parse_bool_env("ENABLE_DEMO_AUTOLOGIN", default=not IS_PRODUCTION)
ENABLE_NETWORK_SNIFFING = parse_bool_env("ENABLE_NETWORK_SNIFFING", default=not IS_PRODUCTION)
ENABLE_ENDPOINT_SIMULATION = parse_bool_env("ENABLE_ENDPOINT_SIMULATION", default=not IS_PRODUCTION)
RATES_LIMIT_ENABLED = parse_bool_env("RATE_LIMIT_ENABLED", default=IS_PRODUCTION)
RATE_LIMIT_PER_MINUTE = parse_int_env("RATE_LIMIT_PER_MINUTE", 180)
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
            self.client.admin.command("ping")
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
                "timestamp": datetime.now(),
                "event_type": event_type,
                "description": description,
                "user_id": user_id,
                "severity": severity,
                "ip_address": request.remote_addr if request else None
            }
            if hasattr(self.system_logs, "insert_one"):
                self.system_logs.insert_one(log_entry)
            elif isinstance(self.system_logs, list):
                self.system_logs.append(log_entry)
        except Exception as e:
            logging.error(f"Failed to log system event: {e}")

    def create_user(self, username, email, password_hash):
        """Create a new user in MongoDB"""
        try:
            user_data = {
                "username": username,
                "email": email,
                "password": password_hash,
                "created_at": datetime.now(),
                "last_login": None,
                "is_active": True
            }
            if hasattr(self.users, "insert_one"):
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
            if hasattr(self.users, "find_one"):
                return self.users.find_one({"username": username})
            if isinstance(self.users, dict):
                return self.users.get(username)
            return None
        except Exception as e:
            logging.error(f"Failed to get user: {e}")
            return None

    def update_user_login(self, username):
        """Update user's last login timestamp"""
        try:
            if hasattr(self.users, "update_one"):
                self.users.update_one(
                    {"username": username},
                    {"$set": {"last_login": datetime.now()}}
                )
            elif isinstance(self.users, dict) and username in self.users:
                self.users[username]["last_login"] = datetime.now()
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
    secret_key = os.getenv("SECRET_KEY")
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
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = IS_PRODUCTION
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(seconds=parse_int_env("SESSION_LIFETIME_SECONDS", 3600))
socketio = SocketIO(app, cors_allowed_origins=CORS_ALLOWED_ORIGINS)

network_sniffing_started = False
network_sniffing_lock = threading.Lock() # Corrected typo from etwork_sniffing_lock

# Auto-login user for demo purposes
@app.before_request
def auto_login():
    """Automatically log in user for demo - remove in production"""
    if not ENABLE_DEMO_AUTOLOGIN:
        return
    if "user_id" not in session:
        session["user_id"] = "demo_user"
        session["user_email"] = "demo@malwaredetection.com"
        mongo_db.log_system_event("AUTO_LOGIN", "User automatically logged in for demo", "demo_user", "INFO")


@app.before_request
def basic_rate_limit_guard():
    if not RATES_LIMIT_ENABLED:
        return None

    path = request.path or "/"
    if path.startswith("/static/") or path.startswith("/socket.io"):
        return None

    now = time.time()
    src_ip = request.headers.get("X-Forwarded-For", request.remote_addr) or "unknown"
    if isinstance(src_ip, str) and "," in src_ip:
        src_ip = src_ip.split(",")[0].strip()

    with _rate_limit_lock:
        timestamps = [ts for ts in _rate_limit_cache[src_ip] if now - ts < 60]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            if path.startswith("/api"):
                return jsonify({"status": "error", "message": "Rate limit exceeded"}), 429
            return "Rate limit exceeded", 429
        timestamps.append(now)
        _rate_limit_cache[src_ip] = timestamps
    return None

# ----------------- Updated Routes with Consistent Names ----------------
@app.route("/")
def home():
    username = session.get("user_id")
    return render_template("index.html", username=username)


@app.route("/api_analyzer", methods=["GET"])
@app.route("/api-analyzer", methods=["GET"])
def api_analyzer():
    username = session.get("user_id")
    return render_template("api_analyzer.html", username=username)

@app.route("/file", methods=["GET"])
def file():
    username = session.get("user_id")
    return render_template("file.html", username=username)

@app.route("/network", methods=["GET"])
def network():
    global network_sniffing_started
    username = session.get("user_id")

    if ENABLE_NETWORK_SNIFFING:
        with network_sniffing_lock:
            if not network_sniffing_started:
                sniff_thread = threading.Thread(target=start_sniffing)
                sniff_thread.daemon = True
                sniff_thread.start()
                network_sniffing_started = True
    return render_template("network.html", username=username)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok", "env": APP_ENV})

@app.route("/restricted", methods=["GET"])
def restricted():
    username = session.get("user_id")
    return render_template("restricted.html", username=username)

@app.route("/about", methods=["GET"])
def about():
    username = session.get("user_id")
    return render_template("about.html", username=username)

# Simple logout for demo
@app.route("/logout")
def logout():
    username = session.get("user_id")
    session.clear()
    mongo_db.log_system_event("USER_LOGOUT", f"User {username} logged out", username, "INFO")
    flash("You have been logged out successfully", "success")
    # Auto-login again for demo
    session["user_id"] = "demo_user"
    session["user_email"] = "demo@malwaredetection.com"
    return redirect(url_for("home"))


# ----------------- Authentication Routes ----------------
@app.route("/signin", methods=["GET", "POST"])
def signin():
    """User sign in route"""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Simple authentication for demo
        if username and password:
            session["user_id"] = username
            session["user_email"] = f"{username}@malwaredetection.com"
            mongo_db.log_system_event("USER_LOGIN", f"User {username} logged in", username, "INFO")
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Please enter both username and password", "error")

    return render_template("signin.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """User sign up route"""
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # Simple validation for demo
        if password != confirm_password:
            flash("Passwords do not match", "error")
        elif username and email and password:
            # For demo purposes, just create session
            session["user_id"] = username
            session["user_email"] = email
            mongo_db.log_system_event("USER_SIGNUP", f"New user {username} registered", username, "INFO")
            flash("Registration successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Please fill all fields", "error")

    return render_template("signup.html")

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


# ----------------- API Behavior Prediction - Start of New Code ----------------

# Model and LabelEncoder paths for API behavior prediction
API_BEHAVIOR_MODEL_SAVE_DIR = '/content/drive/MyDrive/problem solved'
API_BEHAVIOR_MODEL_FILENAME = 'Random_Forest_robust_source.pkl' # Chosen best performing model with robust source
LE_SOURCE_ROBUST_FILENAME = 'le_source_robust.pkl'
LE_TARGET_FILENAME = 'label_encoder.pkl'

# Full paths for loading
API_BEHAVIOR_MODEL_PATH = os.path.join(API_BEHAVIOR_MODEL_SAVE_DIR, API_BEHAVIOR_MODEL_FILENAME)
API_LE_SOURCE_ROBUST_PATH = os.path.join(API_BEHAVIOR_MODEL_SAVE_DIR, LE_SOURCE_ROBUST_FILENAME)
API_LE_TARGET_PATH = os.path.join(API_BEHAVIOR_MODEL_SAVE_DIR, LE_TARGET_FILENAME)

# Hardcoded training means for numerical imputation (from notebook analysis)
API_TRAIN_MEAN_INTER_API_ACCESS_DURATION = 20.548672
API_TRAIN_MEAN_API_ACCESS_UNIQUENESS = 0.444437

# Hardcoded unique categories for other LabelEncoders (from notebook analysis)
API_IP_TYPE_CLASSES = ['default', 'private_ip', 'datacenter', 'google_bot', 'unknown_ip_type']
API_BEHAVIOR_TYPE_CLASSES = ['outlier', 'normal', 'bot', 'attack', 'unknown_behavior_type']
API_BEHAVIOR_CLASSES = [
    'outlier', 'Normal', 'Googlebot/2.1', 'BingPreview/1.0b', 'SQL Injection',
    'AdsBot-Google', 'unknown_behavior_category'
]

# Load LabelEncoders for API behavior prediction
api_le_source_robust = safe_joblib_load(API_LE_SOURCE_ROBUST_PATH, 'API behavior robust source LabelEncoder')
api_le_target = safe_joblib_load(API_LE_TARGET_PATH, 'API behavior target LabelEncoder')

# Initialize and fit other LabelEncoders using hardcoded classes (for features not explicitly saved)
api_le_ip_type = LabelEncoder().fit(API_IP_TYPE_CLASSES)
api_le_behavior_type = LabelEncoder().fit(API_BEHAVIOR_TYPE_CLASSES)
api_le_behavior = LabelEncoder().fit(API_BEHAVIOR_CLASSES)

# Load the ML Model for API behavior prediction
api_prediction_model = safe_joblib_load(API_BEHAVIOR_MODEL_PATH, 'API behavior prediction model')
if api_prediction_model is None:
    logging.error("API behavior prediction ML model not loaded; prediction endpoint will primarily rely on rule-based classification.")


def classify_api_call(row: pd.Series):
    """
    Rule-based classification for API calls based on predefined criteria.
    Returns 'attack', 'bot', 'normal', or 'outlier'.
    """
    # Extract relevant feature values safely, providing defaults if keys might be missing
    ip_type = row.get('ip_type', 'unknown_ip_type')
    behavior_type = row.get('behavior_type', 'unknown_behavior_type')
    behavior = row.get('behavior', 'unknown_behavior_category')
    inter_api_access_duration_sec = row.get('inter_api_access_duration(sec)', API_TRAIN_MEAN_INTER_API_ACCESS_DURATION)
    api_access_uniqueness = row.get('api_access_uniqueness', API_TRAIN_MEAN_API_ACCESS_UNIQUENESS)
    sequence_length_count = row.get('sequence_length(count)', 0)
    vsession_duration_min = row.get('vsession_duration(min)', 0)
    num_sessions = row.get('num_sessions', 0)
    num_users = row.get('num_users', 0)
    num_unique_apis = row.get('num_unique_apis', 0)
    source = row.get('source', 'unknown_source_category')

    # 1. Attack classification (most aggressive, check first)
    if (behavior_type == 'attack' or
            'SQL Injection' in behavior or
            sequence_length_count > 100 or
            vsession_duration_min > 100000 or
            num_sessions > 100 or
            num_users > 50):
        return 'attack'

    # 2. Bot classification
    if (behavior_type == 'bot' or
            ip_type == 'google_bot' or
            ip_type == 'datacenter' or
            ip_type == 'private_ip' or
            'Googlebot' in behavior or
            'BingPreview' in behavior or
            'AdsBot-Google' in behavior or
            source == 'E'):
        return 'bot'

    # Additional bot conditions for 'outlier' behavior_type that exhibits bot-like numerical patterns
    if (behavior_type == 'outlier' and
            inter_api_access_duration_sec < 0.01 and
            (sequence_length_count > 100 or num_sessions > 10) and
            (vsession_duration_min > 30000 or num_users > 10 or num_unique_apis > 8)):
        return 'bot'

    # 3. Normal classification
    if (behavior_type == 'normal' and
            ip_type == 'default' and
            0.38 <= inter_api_access_duration_sec <= 9.49 and
            0.19 <= api_access_uniqueness <= 0.66 and
            6.66 <= sequence_length_count <= 58 and
            543 <= vsession_duration_min <= 25056 and
            1 <= num_sessions <= 3 and
            1 <= num_users <= 2 and
            5 <= num_unique_apis <= 21):
        return 'normal'

    # 4. Outlier classification (if none of the above specific conditions are met)
    return 'outlier'


def safe_label_transform(encoder: LabelEncoder, value, unknown_placeholder_str: str) -> int:
    """
    Safely transforms a categorical value using a LabelEncoder, handling unseen values
    by mapping them to a predefined 'unknown' placeholder that the encoder was fitted on.
    """
    if pd.isna(value) or value is None:
        value_to_transform = unknown_placeholder_str
    else:
        value_to_transform = str(value) # Ensure value is string for encoder

    if value_to_transform not in encoder.classes_:
        value_to_transform = unknown_placeholder_str

    try:
        return encoder.transform([value_to_transform])[0]
    except Exception as e:
        logging.error(f"Error transforming '{value_to_transform}' with encoder {encoder}. Falling back to 0: {e}")
        return 0 # Fallback to 0 if transformation fails


def preprocess_api_data(raw_data: dict, model_expects_source_type: bool) -> pd.DataFrame:
    """
    Preprocesses raw API call data into a DataFrame suitable for ML model prediction.
    Ensures consistency with training data preprocessing.
    """
    # Convert raw_data to DataFrame, handling missing keys by adding them with NaN
    input_df = pd.DataFrame([raw_data])

    # Ensure all expected raw columns exist, filling with NaN if not present
    expected_raw_cols = [
        'inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
        'vsession_duration(min)', 'ip_type', 'behavior', 'behavior_type', 'num_sessions',
        'num_users', 'num_unique_apis', 'source'
    ]
    for col in expected_raw_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # 1. Drop identifier columns if they exist (not expected from raw_data dict but for robustness)
    cols_to_drop = ['Unnamed: 0', '_id']
    input_df = input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns])

    # 2. Fill missing numerical values with training means (for main numerical features)
    input_df['inter_api_access_duration(sec)'] = input_df['inter_api_access_duration(sec)'].fillna(API_TRAIN_MEAN_INTER_API_ACCESS_DURATION)
    input_df['api_access_uniqueness'] = input_df['api_access_uniqueness'].fillna(API_TRAIN_MEAN_API_ACCESS_UNIQUENESS)
    # Fill other numerical NaNs with 0 (or a more suitable default from training data analysis)
    input_df['sequence_length(count)'] = input_df['sequence_length(count)'].fillna(0)
    input_df['vsession_duration(min)'] = input_df['vsession_duration(min)'].fillna(0)
    input_df['num_sessions'] = input_df['num_sessions'].fillna(0)
    input_df['num_users'] = input_df['num_users'].fillna(0)
    input_df['num_unique_apis'] = input_df['num_unique_apis'].fillna(0)

    # 3. Apply robust LabelEncoding for 'source'
    if api_le_source_robust is not None:
        input_df['source_type'] = input_df['source'].apply(lambda x: safe_label_transform(api_le_source_robust, x, 'unknown_source_category'))
    else:
        logging.warning("api_le_source_robust not loaded. Defaulting 'source_type' to 0.")
        input_df['source_type'] = 0 # Fallback if encoder not loaded

    # 4. Apply LabelEncoding for other categorical features
    if api_le_ip_type is not None:
        input_df['type_ip'] = input_df['ip_type'].apply(lambda x: safe_label_transform(api_le_ip_type, x, 'unknown_ip_type'))
    else:
        logging.warning("api_le_ip_type not loaded. Defaulting 'type_ip' to 0.")
        input_df['type_ip'] = 0

    if api_le_behavior_type is not None:
        input_df['type_behaviour'] = input_df['behavior_type'].apply(lambda x: safe_label_transform(api_le_behavior_type, x, 'unknown_behavior_type'))
    else:
        logging.warning("api_le_behavior_type not loaded. Defaulting 'type_behaviour' to 0.")
        input_df['type_behaviour'] = 0

    if api_le_behavior is not None:
        input_df['behaviour'] = input_df['behavior'].apply(lambda x: safe_label_transform(api_le_behavior, x, 'unknown_behavior_category'))
    else:
        logging.warning("api_le_behavior not loaded. Defaulting 'behaviour' to 0.")
        input_df['behaviour'] = 0

    # 5. Select and order features consistently with training data
    if model_expects_source_type:
        feature_cols = ['inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
                        'vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis',
                        'type_ip', 'type_behaviour', 'source_type', 'behaviour']
    else:
        feature_cols = ['inter_api_access_duration(sec)', 'api_access_uniqueness', 'sequence_length(count)',
                        'vsession_duration(min)', 'num_sessions', 'num_users', 'num_unique_apis',
                        'type_ip', 'type_behaviour', 'behaviour']

    # Ensure all feature columns exist in the DataFrame and are in the correct order
    final_processed_df = input_df[feature_cols]

    return final_processed_df

@app.route('/predict_behavior', methods=['POST'])
def predict_behavior():
    """
    API endpoint to receive raw API call data, preprocess it, and predict its behavior.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    raw_api_data = request.get_json()

    # This boolean should reflect how the currently loaded ML model was trained.
    # We assume 'Random_Forest_robust_source.pkl' was chosen, which includes 'source_type'.
    model_expects_source_type = True

    try:
        # Preprocess the incoming data
        processed_data_df = preprocess_api_data(raw_api_data, model_expects_source_type)

        if api_prediction_model is None or api_le_target is None:
            logging.warning("ML model or target LabelEncoder not loaded for /predict_behavior. Falling back to rule-based classification.")
            classified_behavior = classify_api_call(pd.Series(raw_api_data)) # Pass raw_api_data as Series for classify_api_call
            return jsonify({
                "predicted_behavior": classified_behavior,
                "method": "rule_based_fallback",
                "message": "ML model or LabelEncoder not loaded, falling back to rule-based classification."
            })

        # Make prediction using the loaded ML model
        prediction_encoded = api_prediction_model.predict(processed_data_df)

        # Inverse transform the prediction to human-readable label
        predicted_label = api_le_target.inverse_transform(prediction_encoded)[0]

        return jsonify({
            "predicted_behavior": predicted_label,
            "method": "machine_learning",
            "input_data_received": raw_api_data # Optionally return input data for debugging
        })

    except Exception as e:
        logging.exception(f"Error during API behavior prediction: {e}")
        # Fallback to rule-based classification on error as well
        classified_behavior = classify_api_call(pd.Series(raw_api_data)) # Pass raw_api_data as Series
        return jsonify({
            "error": str(e),
            "predicted_behavior_fallback": classified_behavior,
            "method": "error_fallback_to_rule_based",
            "message": "An error occurred during ML prediction, falling back to rule-based classification."
        }), 500

# ----------------- API Behavior Prediction - End of New Code ----------------


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
        # For a Flask app, it's typical to just start the thread and let it run.
        # The main app thread will continue to serve requests.

    except Exception as e:
        logging.error(f"Error starting sniffing threads: {e}")

# Flask application entry point
if __name__ == '__main__':
    # Ensure any necessary directories are created or models loaded before starting the app
    # For instance, if you want to run the network sniffing simulation automatically on startup:
    # with network_sniffing_lock:
    #     if not network_sniffing_started and ENABLE_NETWORK_SNIFFING:
    #         sniff_thread = threading.Thread(target=start_sniffing)
    #         sniff_thread.daemon = True
    #         sniff_thread.start()
    #         network_sniffing_started = True

    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
