"""
Production runner for Coconut Leaf Disease Predictor
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after loading environment variables
from app import app, init_db, logger
import config

def setup_production():
    """Setup production environment"""
    
    # Validate critical settings
    if config.SECRET_KEY == 'your-secret-key-change-this-in-production':
        logger.error("CRITICAL: Default SECRET_KEY detected! Change it before running in production.")
        sys.exit(1)
    
    if config.DEBUG and config.ENVIRONMENT == 'production':
        logger.warning("WARNING: DEBUG mode is enabled in production environment!")
    
    # Initialize database
    init_db()
    
    # Create upload folder
    if not os.path.exists(config.UPLOAD_FOLDER):
        os.makedirs(config.UPLOAD_FOLDER)
        logger.info(f"Created upload folder: {config.UPLOAD_FOLDER}")
    
    logger.info("Production setup completed successfully")

if __name__ == '__main__':
    setup_production()
    
    # Run with Gunicorn in production, Flask dev server otherwise
    if config.ENVIRONMENT == 'production':
        logger.info("Starting production server with Gunicorn...")
        os.system(f"gunicorn --bind {config.HOST}:{config.PORT} --workers 4 wsgi:app")
    else:
        logger.info("Starting development server...")
        app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
