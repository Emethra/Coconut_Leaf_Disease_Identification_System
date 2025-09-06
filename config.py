# Configuration file for Coconut Leaf Disease Predictor
import os
from datetime import timedelta

# Application Settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-this-in-production')
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = int(os.environ.get('PORT', 5000))

# Upload Settings
UPLOAD_FOLDER = 'uploads'
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Database Settings
DATABASE_NAME = 'coconut_leaf.db'

# Model Settings
MODEL_PATH = 'coconut_leaf_disease_model.h5'
IMAGE_SIZE = (224, 224)  # Input size for the model
DISEASE_CLASSES = [
    'Healthy',
    'leaf_blast',
    'leaf_spot'
    
]

# Security Settings (for production)
SECURE_COOKIES = os.environ.get('SECURE_COOKIES', 'False').lower() == 'true'
SESSION_TIMEOUT = int(os.environ.get('SESSION_TIMEOUT', 3600))  # Session timeout in seconds
PERMANENT_SESSION_LIFETIME = timedelta(seconds=SESSION_TIMEOUT)

# Production Settings
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

# Rate Limiting (requests per minute)
RATE_LIMIT_PREDICTION = int(os.environ.get('RATE_LIMIT_PREDICTION', 10))
RATE_LIMIT_GENERAL = int(os.environ.get('RATE_LIMIT_GENERAL', 100))
