import pandas as pd
import numpy as np
import joblib
import psutil
import socket
import subprocess
import platform
from datetime import datetime
import re
import warnings
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RealTimeDataCollector:
    """Collect real-time system and network data for attack detection"""

    def __init__(self):
        self.system_info = self.get_system_info()

    def get_system_info(self):
        """Get basic system information"""
        return {
            'hostname': socket.gethostname(),
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0]
        }

    def get_network_connections(self):
        """Get current network connections"""
        connections = []
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    connections.append({
                        'local_address': f"{conn.laddr.ip}:{conn.laddr.port}",
                        'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else '',
                        'status': conn.status,
                        'pid': conn.pid
                    })
        except Exception as e:
            logging.error(f"Error getting network connections: {e}")
        return connections

    def get_process_info(self, pid):
        """Get information about a specific process"""
        try:
            process = psutil.Process(pid)
            return {
                'pid': pid,
                'name': process.name(),
                'exe': process.exe(),
                'cmdline': ' '.join(process.cmdline()) if process.cmdline() else '',
                'username': process.username(),
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent()
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def get_active_connections_with_processes(self):
        """Get active network connections with process information"""
        active_connections = []
        connections = self.get_network_connections()

        for conn in connections[:10]:  # Limit to first 10 connections
            if conn['pid']:
                process_info = self.get_process_info(conn['pid'])
                if process_info:
                    connection_data = {
                        **conn,
                        'process_name': process_info['name'],
                        'process_cmdline': process_info['cmdline'],
                        'remote_ip': conn['remote_address'].split(':')[0] if ':' in conn['remote_address'] else '',
                        'remote_port': conn['remote_address'].split(':')[1] if ':' in conn['remote_address'] else ''
                    }
                    active_connections.append(connection_data)

        return active_connections

    def get_dns_queries(self):
        """Get recent DNS queries (varies by OS)"""
        dns_queries = []
        try:
            if platform.system() == "Windows":
                # Windows DNS cache
                result = subprocess.run(['ipconfig', '/displaydns'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Record Name' in line:
                        domain = line.split(':')[-1].strip()
                        if domain and domain != 'localhost':
                            dns_queries.append(domain)
            elif platform.system() == "Linux":
                # Linux - check /etc/hosts and recent connections
                try:
                    with open('/etc/hosts', 'r') as f:
                        for line in f:
                            if not line.startswith('#') and 'localhost' not in line:
                                parts = line.split()
                                if len(parts) > 1:
                                    dns_queries.extend(parts[1:])
                except:
                    pass
        except Exception as e:
            logging.error(f"Error getting DNS queries: {e}")

        return list(set(dns_queries))[:20]  # Return unique domains, limit to 20

    def get_system_activity(self):
        """Get current system activity metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'network_io': psutil.net_io_counters()._asdict(),
            'disk_io': psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
            'active_processes': len(psutil.pids())
        }


class RestrictedSiteAccessDetector:
    def __init__(self):
        """
        Initialize the Restricted Site Access Detection system using individual model files.
        """
        self.model = None
        self.label_encoders = None
        self.target_encoder = None
        self.data_collector = RealTimeDataCollector()
        self.feature_columns = [
            'time', 'end_time', 'user_id', 'source_ip', 'domain', 'domain_type',
            'access_type', 'request_type', 'protocol', 'vpn_usage', 'tor_usage',
            'dns_encryption', 'user_role', 'user_activity_type', 'recent_web_access_attempts',
            'status', 'attack_risk_level'
        ]
        self.model_loaded = False
        self.load_models()

    def load_models(self):
        """Load the individual pre-trained models and encoders"""
        try:
            # Load the main model
            logging.info("Loading attack prediction model...")
            self.model = joblib.load('restricted/attack_prediction_model.pkl')
            logging.info("✅ Attack prediction model loaded successfully!")

            # Load label encoders
            logging.info("Loading label encoders...")
            self.label_encoders = joblib.load('restricted/label_encoders.pkl')
            logging.info("✅ Label encoders loaded successfully!")

            # Load target encoder
            logging.info("Loading target encoder...")
            self.target_encoder = joblib.load('restricted/target_encoder.pkl')
            logging.info("✅ Target encoder loaded successfully!")

            self.model_loaded = True
            logging.info("🎉 All models loaded successfully - ML prediction enabled!")

        except Exception as e:
            logging.warning(f"❌ Error loading models: {e}")
            logging.info("🔄 Falling back to rule-based detection")
            self.model_loaded = False
            self.initialize_fallback_encoders()

    def initialize_fallback_encoders(self):
        """Initialize fallback encoders when models can't be loaded"""
        from sklearn.preprocessing import LabelEncoder

        # Create default label encoders
        self.label_encoders = {}
        for col in ['domain', 'domain_type', 'access_type', 'request_type', 'protocol',
                    'user_role', 'user_activity_type', 'status', 'attack_risk_level']:
            le = LabelEncoder()
            if col == 'domain_type':
                le.fit(['Social Media Site', 'Educational Website', 'Suspicious Site', 'Unknown'])
            elif col == 'access_type':
                le.fit(['Allowed', 'Restricted', 'Suspicious'])
            elif col == 'attack_risk_level':
                le.fit(['Low', 'Medium', 'High'])
            else:
                le.fit(['Unknown'])
            self.label_encoders[col] = le

        # Create default target encoder
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(['Benign', 'Ransomware', 'Phishing', 'Distributed Denial of Service (DDoS)', 'Fraud'])

    def time_to_minutes_safe(self, time_val):
        """Convert time to minutes (same as in training)"""
        if isinstance(time_val, str):
            try:
                time_obj = pd.to_datetime(time_val, format='%H:%M:%S')
                return time_obj.hour * 60 + time_obj.minute
            except:
                return 0
        elif hasattr(time_val, 'hour') and hasattr(time_val, 'minute'):
            return time_val.hour * 60 + time_val.minute
        elif isinstance(time_val, (int, float)):
            return time_val
        else:
            return 0

    def collect_real_time_data(self):
        """Collect real-time data from the system"""
        logging.info("🔄 Collecting real-time system data...")

        # Get active network connections
        active_connections = self.data_collector.get_active_connections_with_processes()
        dns_queries = self.data_collector.get_dns_queries()
        system_activity = self.data_collector.get_system_activity()

        real_time_data = []

        for conn in active_connections:
            # Analyze each connection
            domain = self.extract_domain_from_connection(conn)

            # Create feature set for prediction
            features = {
                'time': self.time_to_minutes_safe(datetime.now().strftime('%H:%M:%S')),
                'end_time': self.time_to_minutes_safe((datetime.now().timestamp() + 3600).strftime('%H:%M:%S')),
                'user_id': f"user_{conn.get('pid', 'unknown')}",
                'source_ip': socket.gethostbyname(socket.gethostname()),
                'domain': domain,
                'domain_type': self.classify_domain_type(domain),
                'access_type': self.determine_access_type(conn),
                'request_type': 'GET',  # Default assumption
                'protocol': self.determine_protocol(conn),
                'vpn_usage': self.check_vpn_usage(),
                'tor_usage': self.check_tor_usage(conn),
                'dns_encryption': 0,  # Default assumption
                'user_role': self.determine_user_role(conn),
                'user_activity_type': self.determine_activity_type(conn),
                'recent_web_access_attempts': len(active_connections),
                'status': 'Successful',
                'attack_risk_level': self.assess_risk_level(conn, domain)
            }

            real_time_data.append({
                'connection_info': conn,
                'features': features,
                'system_metrics': system_activity
            })

        return real_time_data

    def extract_domain_from_connection(self, conn):
        """Extract domain from connection information"""
        remote_ip = conn.get('remote_ip', '')

        # Try to resolve IP to domain
        try:
            if remote_ip and not remote_ip.startswith(('127.', '192.168.', '10.')):
                domain = socket.gethostbyaddr(remote_ip)[0]
                return domain
        except:
            pass

        # Check process command line for domains
        cmdline = conn.get('process_cmdline', '').lower()
        domain_pattern = r'([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
        domains = re.findall(domain_pattern, cmdline)
        if domains:
            return domains[0][0]

        return remote_ip  # Return IP if no domain found

    def classify_domain_type(self, domain):
        """Classify domain type based on patterns"""
        if not domain or domain.replace('.', '').isdigit():
            return 'Unknown'

        domain_lower = domain.lower()

        # Social Media
        social_media_keywords = ['facebook', 'twitter', 'instagram', 'linkedin', 'youtube', 'tiktok', 'threads']
        if any(keyword in domain_lower for keyword in social_media_keywords):
            return 'Social Media Site'

        # Search Engines
        search_engines = ['google', 'bing', 'yahoo', 'duckduckgo']
        if any(engine in domain_lower for engine in search_engines):
            return 'Search Engine'

        # Educational
        educational = ['edu', 'academy', 'course', 'university', 'college']
        if any(edu in domain_lower for edu in educational):
            return 'Educational Website'

        # Suspicious patterns
        suspicious = ['tor', 'proxy', 'vpn', 'anonymous', 'free', 'crack', 'keygen']
        if any(susp in domain_lower for susp in suspicious):
            return 'Suspicious Site'

        return 'General Website'

    def determine_access_type(self, conn):
        """Determine access type based on connection and process"""
        process_name = conn.get('process_name', '').lower()
        domain = self.extract_domain_from_connection(conn)
        domain_type = self.classify_domain_type(domain)

        if domain_type in ['Suspicious Site', 'Unknown']:
            return 'Suspicious'
        elif domain_type in ['Social Media Site']:
            return 'Restricted'
        else:
            return 'Allowed'

    def determine_protocol(self, conn):
        """Determine protocol based on port and process"""
        remote_port = conn.get('remote_port', '')
        process_name = conn.get('process_name', '').lower()

        if remote_port in ['80', '8080']:
            return 'HTTP'
        elif remote_port in ['443', '8443']:
            return 'HTTPS'
        elif remote_port in ['22']:
            return 'SSH'
        elif remote_port in ['21']:
            return 'FTP'
        else:
            return 'Other'

    def check_vpn_usage(self):
        """Check if VPN is being used"""
        try:
            # Check for common VPN processes
            vpn_processes = ['openvpn', 'wireguard', 'expressvpn', 'nordvpn', 'protonvpn']
            for process in psutil.process_iter(['name']):
                if any(vpn in process.info['name'].lower() for vpn in vpn_processes):
                    return 1
        except:
            pass
        return 0

    def check_tor_usage(self, conn):
        """Check for Tor usage"""
        process_name = conn.get('process_name', '').lower()
        remote_ip = conn.get('remote_ip', '')

        # Check for Tor process
        if 'tor' in process_name:
            return 1

        # Check for known Tor nodes (simplified)
        if remote_ip.startswith('1.2.3.'):  # Example Tor IP range
            return 1

        return 0

    def determine_user_role(self, conn):
        """Determine user role based on process and behavior"""
        process_name = conn.get('process_name', '').lower()
        username = conn.get('username', '')

        if 'admin' in username.lower() or 'root' in username.lower():
            return 'Admin'
        elif any(proc in process_name for proc in ['chrome', 'firefox', 'edge', 'safari']):
            return 'Staff'
        else:
            return 'Guest'

    def determine_activity_type(self, conn):
        """Determine activity type based on process and connection"""
        process_name = conn.get('process_name', '').lower()
        remote_port = conn.get('remote_port', '')

        if any(proc in process_name for proc in ['chrome', 'firefox', 'edge', 'safari']):
            return 'Browsing'
        elif remote_port in ['21', '22', '445']:
            return 'File Access'
        elif any(proc in process_name for proc in ['outlook', 'thunderbird']):
            return 'Email'
        else:
            return 'System Activity'

    def assess_risk_level(self, conn, domain):
        """Assess risk level based on multiple factors"""
        risk_score = 0

        # Domain type risk
        domain_type = self.classify_domain_type(domain)
        if domain_type == 'Suspicious Site':
            risk_score += 3
        elif domain_type == 'Unknown':
            risk_score += 2
        elif domain_type == 'Social Media Site':
            risk_score += 1

        # Process risk
        process_name = conn.get('process_name', '').lower()
        suspicious_processes = ['tor', 'proxy', 'vpn', 'cmd', 'powershell', 'terminal']
        if any(proc in process_name for proc in suspicious_processes):
            risk_score += 2

        # Port risk
        remote_port = conn.get('remote_port', '')
        risky_ports = ['22', '23', '135', '445', '1433', '3389']
        if remote_port in risky_ports:
            risk_score += 2

        # Determine risk level
        if risk_score >= 4:
            return 'High'
        elif risk_score >= 2:
            return 'Medium'
        else:
            return 'Low'

    def preprocess_features(self, input_data):
        """Preprocess input features to match training format"""
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()

        # Ensure all required columns are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value

        # Convert time columns
        df['time'] = df['time'].apply(self.time_to_minutes_safe)
        df['end_time'] = df['end_time'].apply(self.time_to_minutes_safe)

        # Encode categorical variables
        for col in ['domain', 'domain_type', 'access_type', 'request_type', 'protocol',
                    'user_role', 'user_activity_type', 'status', 'attack_risk_level']:
            if col in df.columns and col in self.label_encoders:
                try:
                    # Handle unseen labels
                    current_values = df[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)

                    # Replace unseen values with the first known class
                    unseen_mask = ~current_values.isin(known_classes)
                    if unseen_mask.any():
                        default_value = self.label_encoders[col].classes_[0]
                        current_values[unseen_mask] = default_value

                    df[col] = self.label_encoders[col].transform(current_values)
                except Exception as e:
                    logging.warning(f"Encoding error for {col}: {e}")
                    df[col] = 0

        return df[self.feature_columns]

    def rule_based_prediction(self, features):
        """Fallback rule-based prediction when ML model is not available"""
        domain = features.get('domain', '')
        domain_type = features.get('domain_type', '')
        access_type = features.get('access_type', '')
        attack_risk_level = features.get('attack_risk_level', 'Low')

        # Simple rule-based logic
        if domain_type == 'Suspicious Site' or access_type == 'Suspicious':
            if attack_risk_level == 'High':
                return {
                    'prediction': 'Ransomware',
                    'confidence': 0.85,
                    'is_malicious': True
                }
            else:
                return {
                    'prediction': 'Phishing',
                    'confidence': 0.75,
                    'is_malicious': True
                }
        elif domain_type == 'Social Media Site' and attack_risk_level == 'High':
            return {
                'prediction': 'Fraud',
                'confidence': 0.65,
                'is_malicious': True
            }
        else:
            return {
                'prediction': 'Benign',
                'confidence': 0.90,
                'is_malicious': False
            }

    def predict_single_connection(self, data_point):
        """Predict attack type for a single connection"""
        try:
            # Preprocess features
            processed_data = self.preprocess_features(data_point['features'])

            if self.model_loaded and self.model is not None:
                # Use ML model for prediction
                prediction_encoded = self.model.predict(processed_data)[0]
                probabilities = self.model.predict_proba(processed_data)[0]

                # Decode prediction using target encoder
                prediction_decoded = self.target_encoder.inverse_transform([prediction_encoded])[0]

                # Get class names and probabilities
                class_names = self.target_encoder.classes_

                result = {
                    'prediction': {
                        'attack_type': prediction_decoded,
                        'probabilities': {
                            class_name: float(prob)
                            for class_name, prob in zip(class_names, probabilities)
                        },
                        'is_malicious': prediction_decoded != 'Benign',
                        'confidence': float(np.max(probabilities)),
                        'model_used': 'ML'
                    }
                }
            else:
                # Use rule-based prediction
                rule_result = self.rule_based_prediction(data_point['features'])
                result = {
                    'prediction': {
                        'attack_type': rule_result['prediction'],
                        'probabilities': {
                            rule_result['prediction']: rule_result['confidence'],
                            'Benign': 1 - rule_result['confidence']
                        },
                        'is_malicious': rule_result['is_malicious'],
                        'confidence': rule_result['confidence'],
                        'model_used': 'Rule-Based'
                    }
                }

            return {
                'connection_info': data_point['connection_info'],
                'features': data_point['features'],
                'system_metrics': data_point['system_metrics'],
                'prediction': result['prediction'],
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            # Fallback to basic rule-based prediction
            rule_result = self.rule_based_prediction(data_point['features'])
            return {
                'connection_info': data_point['connection_info'],
                'features': data_point['features'],
                'system_metrics': data_point['system_metrics'],
                'prediction': {
                    'attack_type': rule_result['prediction'],
                    'probabilities': {
                        rule_result['prediction']: rule_result['confidence'],
                        'Benign': 1 - rule_result['confidence']
                    },
                    'is_malicious': rule_result['is_malicious'],
                    'confidence': rule_result['confidence'],
                    'model_used': 'Fallback'
                },
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def predict_real_time_attacks(self):
        """Collect real-time data and predict attacks"""
        logging.info("🎯 Starting real-time attack detection...")

        # Collect real-time data
        real_time_data = self.collect_real_time_data()

        if not real_time_data:
            return {"error": "No active connections found", "model_loaded": self.model_loaded}

        predictions = []

        for data_point in real_time_data:
            prediction_result = self.predict_single_connection(data_point)
            predictions.append(prediction_result)

        return predictions

    def get_system_overview(self):
        """Get overall system overview"""
        active_connections = self.data_collector.get_active_connections_with_processes()
        system_activity = self.data_collector.get_system_activity()
        dns_queries = self.data_collector.get_dns_queries()

        return {
            'system_info': self.data_collector.system_info,
            'active_connections_count': len(active_connections),
            'system_activity': system_activity,
            'recent_dns_queries': dns_queries[:10],  # Last 10 queries
            'collection_time': datetime.now().isoformat(),
            'model_loaded': self.model_loaded
        }


# Global detector instance - initialize only when needed
_detector_instance = None


def get_detector():
    """Get the global detector instance with lazy initialization"""
    global _detector_instance
    if _detector_instance is None:
        try:
            _detector_instance = RestrictedSiteAccessDetector()
        except Exception as e:
            logging.error(f"Failed to initialize detector: {e}")
            # Create a basic instance without model loading
            _detector_instance = RestrictedSiteAccessDetector()
            _detector_instance.model_loaded = False
    return _detector_instance