from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import hashlib
import uuid
import subprocess
import re
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from threading import Thread

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'

# File Paths
users_file = "backend/uploads/user (1).csv"
ratings_file = "backend/uploads/ratings (1).csv"
products_file = "backend/uploads/Skinpro - Skinpro (3).csv"
food_file = "backend/uploads/food (1).csv"
UPLOAD_FOLDER = "../uploads"
YOLO_MODEL_PATH = "backend/uploads/yolo_best3.pt"

# Ensure Uploads Folder Exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure CSV Files Exist
for file, columns in [(users_file, ["user_id", "username", "password", "email", "skin_type"]),
                       (ratings_file, ["user_id", "product", "skin_type", "skin_issues", "ratings"]),
                       (products_file, ["Product", "Concern", "product_url"])]:
    if not os.path.exists(file):
        pd.DataFrame(columns=columns).to_csv(file, index=False)

# Load Data
df_users = pd.read_csv(users_file)
df_ratings = pd.read_csv(ratings_file)
df_products = pd.read_csv(products_file)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_products["Concern"])

# Collaborative Filtering Model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_ratings[['user_id', 'product', 'ratings']], reader)
trainset, _ = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user_id = str(uuid.uuid4())

        # Assign random skin type
        skin_types = ["Oily", "Dry", "Combination", "Sensitive"]
        skin_type = skin_types[len(username) % len(skin_types)]

        new_user = pd.DataFrame([[user_id, username, password, email, skin_type]],
                                columns=df_users.columns)
        df_users.append(new_user, ignore_index=True).to_csv(users_file, index=False)

        return "Registered successfully! <a href='/login'>Login</a>"

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user = df_users[(df_users['email'] == email) & (df_users['password'] == password)]

        if not user.empty:
            session['user_id'] = user.iloc[0]['user_id']
            return redirect(url_for('upload_video'))
        else:
            return "Invalid credentials. <a href='/login'>Try Again</a>"

    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(file_path)
            session['uploaded_file'] = file_path  # Store file path in session
            return redirect(url_for('skin_analysis'))

    return render_template('upload.html')

def run_yolo(video_path):
    """Runs YOLO asynchronously to prevent blocking Flask"""
    yolo_command = f"yolo task=detect mode=predict model={YOLO_MODEL_PATH} conf=0.1 source={video_path} save=True"
    result = subprocess.run(yolo_command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else "Error processing video."

@app.route('/skin_analysis')
def skin_analysis():
    if 'uploaded_file' not in session:
        return "No uploaded file found. <a href='/upload'>Upload Again</a>"

    video_path = session['uploaded_file']
    
    # This will hold the YOLO output
    result_container = {}

    def threaded_yolo():
        result_container['output'] = run_yolo(video_path)

    yolo_thread = Thread(target=threaded_yolo)
    yolo_thread.start()
    yolo_thread.join()

    detected_text = result_container['output']
    detected_labels = re.findall(r"(Acne|Wrinkles|Pigmentation|Sensitive|Normal|Oily|Dry|Combination)", detected_text)

    # ... rest unchanged


@app.route('/recommend')
def recommend_products():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    skin_type = session.get('skin_type', "Unknown")
    concern = " ".join(session.get('skin_issues', ["General"]))

    recommendations = hybrid_recommend(user_id, skin_type, concern)
    return recommendations.to_html()

def hybrid_recommend(user_id, skin_type, concern, top_n=5, w_cbf=0.6, w_cf=0.4):
    input_features = vectorizer.transform([skin_type + " " + concern])
    cbf_scores = cosine_similarity(input_features, tfidf_matrix).flatten()

    cf_scores = np.zeros(len(df_products))
    for idx, pid in enumerate(df_products.index):
        try:
            prediction = model.predict(user_id, int(pid))
            cf_scores[idx] = prediction.est
        except:
            cf_scores[idx] = 3.0

    final_scores = (w_cbf * cbf_scores) + (w_cf * cf_scores)
    top_indices = np.argsort(final_scores)[-top_n:][::-1]

    return df_products.iloc[top_indices][['Product', 'product_url']]

if __name__ == '__main__':
    app.run(debug=True)
