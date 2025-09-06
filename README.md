🌴 Coconut Leaf Disease Identification System
📌 Overview

The Coconut Leaf Disease Identification System is a machine learning–powered web application designed to help farmers and agricultural officers detect coconut leaf diseases early. By uploading an image of a coconut leaf, users can receive an automated prediction of the disease along with confidence scores and recommended treatments.

This project was developed as part of an academic final-year project and aims to bridge the gap between advanced AI research and practical agricultural solutions.

🚀 Features

📷 Image Upload & Prediction – Upload coconut leaf images (JPG, PNG, JPEG, GIF ≤16MB) for real-time analysis.

🤖 Deep Learning Model – CNN model trained using PyTorch and deployed with TensorFlow/Keras.

⚙️ Image Preprocessing – Automatic resizing and normalization for better accuracy.

📊 Prediction Output – Shows disease class, confidence percentage, and CRI-recommended treatment.

👨‍🌾 User Dashboard – Farmers can view their prediction history.

🛠️ Admin Dashboard – Manage users, view statistics, and export prediction data as CSV.

🔐 Secure System – Login/registration with hashed passwords and safe file handling.

🛠️ Tech Stack

Backend: Flask (Python)

Machine Learning: PyTorch, TensorFlow/Keras

Database: SQLite

Frontend: HTML, CSS, JavaScript

Visualization: Matplotlib

Libraries: OpenCV, PIL, Werkzeug

📂 Project Structure
coconut_leaf_disease_identification_system/
│── app.py               # Main Flask app
│── model_train.py        # PyTorch CNN training script
│── inspect_model.py      # TensorFlow/Keras model inspection
│── static/               # Static files (CSS, JS, images)
│── templates/            # HTML templates (Flask views)
│── database.db           # SQLite database
│── leaf_classifier.pth   # Trained PyTorch model
│── coconut_leaf_disease_model.h5 # Deployed Keras model
│── README.md             # Project documentation

⚡ Installation & Usage
🔧 Prerequisites

Python 3.8+

pip (Python package manager)

Git

📥 Clone Repository
git clone https://github.com/Emethra/Coconut_Leaf_Disease_Identification_System.git

cd Coconut_Leaf_Disease_Identification_System

📦 Install Dependencies
pip install -r requirements.txt

▶️ Run the Application
python app.py


Then open your browser at:
👉 http://127.0.0.1:5000

📊 Example Workflow

Register/Login as a user.

Upload a coconut leaf image.

System preprocesses the image and runs prediction using the trained CNN model.

Output: Disease name, confidence percentage, and treatment advice.

Results are stored in your personal dashboard.

Admins can view overall stats and export reports.

🔮 Future Improvements

Mobile app version for direct image capture and prediction.

Multi-disease detection within a single image.

Expand dataset with real-world field images.

Cloud deployment for scalability and real-time analysis.

