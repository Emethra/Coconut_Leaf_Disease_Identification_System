from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import csv
import io
import sqlite3
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
import config

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Simple admin credentials (for demo)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'admin123'  # Change in production

# Allowed file extensions
ALLOWED_EXTENSIONS = config.ALLOWED_EXTENSIONS

# Disease classes
DISEASE_CLASSES = config.DISEASE_CLASSES

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Admin login check
def is_admin():
    return session.get('is_admin')

def init_db():
    """Initialize the SQLite database"""
    conn = sqlite3.connect('coconut_leaf.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_name TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def load_ml_model():
    """Load the trained model"""
    try:
        # Try loading with compile=False to avoid optimizer issues
        model = load_model('coconut_leaf_disease_model.h5', compile=False)
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("The model file may be incompatible with the current TensorFlow version.")
        print("Please retrain the model or use a compatible TensorFlow version.")
        return None

# Load the model at startup
ml_model = load_ml_model()

def preprocess_image(image_file):
    """Preprocess image for prediction"""
    try:
        # Open and resize image
        img = Image.open(image_file)
        img = img.convert('RGB')
        img = img.resize((224, 224))  # Adjust size based on your model requirements
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_disease(image_file):
    """Make prediction on uploaded image"""
    if ml_model is None:
        print("ERROR: ML model is not loaded. Cannot make real predictions.")
        return "Model not available", 0.0

    try:
        # Preprocess image
        processed_image = preprocess_image(image_file)
        if processed_image is None:
            return "Error processing image", 0.0

        # Make prediction using the real ML model
        print("Making prediction with ML model...")
        predictions = ml_model.predict(processed_image, verbose=0)
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])

        # Get class name
        if predicted_class_index < len(DISEASE_CLASSES):
            predicted_class = DISEASE_CLASSES[predicted_class_index]
        else:
            predicted_class = "Unknown"

        print(f"ML Prediction: {predicted_class} with {confidence:.4f} confidence")
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Prediction error", 0.0

@app.route('/')
def home():
    """Home page"""
    return render_template('home.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect('coconut_leaf.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                         (username, email, password_hash))
            conn.commit()
            conn.close()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = sqlite3.connect('coconut_leaf.db')
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, password_hash FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page - available to all users"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return render_template('predict.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return render_template('predict.html')
        
        if file and allowed_file(file.filename):
            try:
                                # Make prediction
                                prediction_result, confidence = predict_disease(file)
                                # Store the latest prediction in session for dashboard use
                                session['last_prediction'] = prediction_result
                                # Save prediction to database if user is logged in
                                if 'user_id' in session:
                                        filename = secure_filename(file.filename)
                                        conn = sqlite3.connect('coconut_leaf.db')
                                        cursor = conn.cursor()
                                        cursor.execute('''INSERT INTO predictions 
                                                                     (user_id, image_name, prediction_result, confidence) 
                                                                     VALUES (?, ?, ?, ?)''',
                                                                 (session['user_id'], filename, prediction_result, confidence))
                                        conn.commit()
                                        conn.close()
                                # Convert image to base64 for display
                                file.seek(0)
                                image_data = base64.b64encode(file.read()).decode()
                                print(f"Rendering predict_result.html with prediction: {prediction_result}, confidence: {confidence}")
                                return render_template('predict_result.html', 
                                                                         prediction=prediction_result,
                                                                         confidence=confidence,
                                                                         image_data=image_data)
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
        else:
            flash('Invalid file type! Please upload an image (PNG, JPG, JPEG, GIF).', 'error')
    
    return render_template('predict.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard - shows prediction history"""
    if 'user_id' not in session:
        flash('Please login to access the dashboard.', 'error')
        return redirect(url_for('login'))
    
    conn = sqlite3.connect('coconut_leaf.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT image_name, prediction_result, confidence, created_at 
                     FROM predictions WHERE user_id = ? 
                     ORDER BY created_at DESC''', (session['user_id'],))
    predictions = cursor.fetchall()
    conn.close()
    
    return render_template('dashboard.html', predictions=predictions)

# Admin login page
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['is_admin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials!', 'error')
    return render_template('admin_login.html')

# Admin dashboard: show user list and delete users
@app.route('/admin/dashboard')
def admin_dashboard():
    # Allow access if admin or normal user is logged in
    if not (is_admin() or 'user_id' in session):
        flash('Login required to access the admin dashboard.', 'error')
        return redirect(url_for('login'))
    
    if is_admin():
        conn = sqlite3.connect('coconut_leaf.db')
        cursor = conn.cursor()
        
        # Get users list
        cursor.execute('SELECT id, username, email, created_at FROM users ORDER BY created_at DESC')
        users = cursor.fetchall()
        
        # Get all predictions with user information
        cursor.execute('''
            SELECT p.id, u.username, p.image_name, p.prediction_result, 
                   p.confidence, p.created_at
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
            LIMIT 50
        ''')
        predictions = cursor.fetchall()
        
        # Get prediction statistics
        cursor.execute('''
            SELECT prediction_result, COUNT(*) as count,
                   AVG(confidence) as avg_confidence
            FROM predictions
            GROUP BY prediction_result
        ''')
        statistics = cursor.fetchall()
        
        conn.close()
        return render_template('admin_dashboard.html', 
                             users=users, 
                             predictions=predictions,
                             statistics=statistics,
                             is_admin=True)
    else:
        # For normal users, show a welcome message only
        return render_template('admin_dashboard.html', 
                             users=None, 
                             predictions=None,
                             statistics=None,
                             is_admin=False)

# Admin delete user
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def admin_delete_user(user_id):
    if not is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))
    conn = sqlite3.connect('coconut_leaf.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin_dashboard'))

# Admin export data
@app.route('/admin/export/<data_type>')
def admin_export_data(data_type):
    """Export data as CSV"""
    if not is_admin():
        flash('Admin access required.', 'error')
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect('coconut_leaf.db')
    cursor = conn.cursor()
    
    if data_type == 'users':
        # Export users data
        cursor.execute('''
            SELECT username, email, created_at
            FROM users
            ORDER BY created_at DESC
        ''')
        rows = cursor.fetchall()
        headers = ['Username', 'Email', 'Created At']
        filename = 'users_export.csv'
    
    elif data_type == 'predictions':
        # Export predictions data with user information
        cursor.execute('''
            SELECT u.username, p.image_name, p.prediction_result,
                   p.confidence, p.created_at
            FROM predictions p
            LEFT JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        ''')
        rows = cursor.fetchall()
        headers = ['Username', 'Image', 'Prediction', 'Confidence', 'Created At']
        filename = 'predictions_export.csv'
    
    else:
        conn.close()
        flash('Invalid export type!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    conn.close()
    
    # Create CSV in memory
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(headers)
    writer.writerows(rows)
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    
    return output

# Admin logout
@app.route('/admin/logout')
def admin_logout():
    session.pop('is_admin', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
