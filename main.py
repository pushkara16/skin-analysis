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
from flask import render_template_string
from threading import Thread
import threading
from ultralytics import YOLO
import torch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
result_cache = {}
# File Paths
users_file = r"C:\Users\Lekhana\Downloads\user (1).csv"
ratings_file = r"C:\Users\Lekhana\Downloads\ratings (1).csv"
products_file = r"C:\Users\Lekhana\Downloads\Skinpro - Skinpro (3).csv"
food_file = r"C:\Users\Lekhana\Downloads\food (1).csv"
UPLOAD_FOLDER = "backend/uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
YOLO_MODEL_A_PATH = r"C:\Users\Lekhana\Downloads\plswork1.pt"
YOLO_MODEL_B_PATH=r"C:\Users\Lekhana\Downloads\tionbest.pt"
model = YOLO(r"C:\Users\Lekhana\Downloads\tionbest.pt")  # load it
# If no GPU, use CPU
# re-export it in current compatible format

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure CSV Files Exist
for file, columns in [(users_file, ["user_id", "username", "password", "email", "skin_type"]),
                       (ratings_file, ["user_id", "product", "skin_type", "skin_issues", "ratings"]),
                       (products_file, ["Product", "Concern", "product_url"])]:
    if not os.path.exists(file):
        pd.DataFrame(columns=columns).to_csv(file, index=False)
df_users = pd.read_csv(users_file)
df_ratings = pd.read_csv(ratings_file)
df_products = pd.read_csv(products_file)
df_food = pd.read_csv(r"C:\Users\Lekhana\Downloads\food (1).csv")
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
@app.route('/get_started')
def get_started():
    return render_template('get_started.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    global df_users
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user_id = str(uuid.uuid4())

        # Assign skin type randomly
        skin_types = ["Oily", "Dry", "Combination", "Sensitive"]
        skin_type = skin_types[len(username) % len(skin_types)]

        new_user = pd.DataFrame([[user_id, username, password, email, skin_type]],
                                columns=df_users.columns)
        df_users = pd.concat([df_users, new_user], ignore_index=True)
        df_users.to_csv(users_file, index=False)

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    global df_users
    if request.method == 'POST':
        email = request.form['email']
        password = hashlib.sha256(request.form['password'].encode()).hexdigest()
        user = df_users[(df_users['email'] == email) & (df_users['password'] == password)]

        if not user.empty:
            session['user_id'] = user.iloc[0]['user_id']
            session['skin_type'] = user.iloc[0]['skin_type']
            return redirect(url_for('upload_video'))
        else:
            return render_template('login.html', error="Invalid credentials.")

    return render_template('login.html')
@app.route('/upload', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
            file.save(file_path)
            session['uploaded_file'] = file_path

            # Start YOLO thread
            thread = Thread(target=process_video, args=(file_path, session['user_id']))
            thread.start()

            return redirect(url_for('skin_analysis'))
    return render_template('ajik.html')
def run_yolo(model_path, conf, video_path, result_dict, key):
    # Explicitly add device=cpu to force CPU usage
    yolo_command = (
        f"yolo task=detect mode=predict model=\"{model_path}\" conf={conf} "
        f"source=\"{video_path}\" save=True device=cpu"
    )
    result = subprocess.run(yolo_command, shell=True, capture_output=True, text=True)
    result_dict[key] = result.stdout if result.returncode == 0 else f"Error: {result.stderr}"

def process_video(video_path, user_id):
    global result_cache
    results = {}

    thread_a = threading.Thread(target=run_yolo, args=(YOLO_MODEL_A_PATH, 0.35, video_path, results, 'model_a'))
    thread_b = threading.Thread(target=run_yolo, args=(YOLO_MODEL_B_PATH, 0.1, video_path, results, 'model_b'))

    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    # Label extraction from both models' output
    labels_a = re.findall(r"(Wrinkles|Normal|Oily|Dry|Combination|Sensitive)", results['model_a'])
    labels_b = re.findall(r"(Acne|Pigmentation|Sensitive)", results['model_b'])

    skin_types = {"Normal", "Oily", "Dry", "Combination"}

    skin_type = next((label for label in labels_a if label in skin_types), "Unknown")
    
    # Assigning labels
    wrinkles = "Wrinkles" if "Wrinkles" in labels_a else None
    acne = "Acne" if "Acne" in labels_b else None
    pigmentation = "Pigmentation" if "Pigmentation" in labels_a or "Pigmentation" in labels_b else None
    sensitivity = "Sensitive" if "Sensitive" in labels_a or "Sensitive" in labels_b else None

# Only include real issues (skip None and Unknown)
    skin_issues = [issue for issue in [wrinkles, acne, pigmentation, sensitivity] if issue is not None]

# Save the results in result_cache and session
    result_cache[user_id] = {
    "skin_type": skin_type,
    "wrinkles": wrinkles,
    "acne": acne,
    "pigmentation": pigmentation,
    "sensitivity": sensitivity,
    "skin_issues": skin_issues,  # Ensure it stores only actual issues
    "done": True
    }


@app.route('/check_analysis_status')
def check_analysis_status():
    user_id = session.get('user_id')
    user_result = result_cache.get(user_id, {})
    return {
        "done": user_result.get('done', False)  # Respond with "done" status
    }

@app.route('/skin_analysis')
def skin_analysis():
    user_id = session.get('user_id')
    user_result = result_cache.get(user_id, {})

    if not user_result.get('done', False):
        return render_template("skin_analysis.html")  # Show loading page

    session.update(user_result)  # Save to session for results page
    return redirect(url_for('show_analysis_result'))


@app.route('/show_analysis_result')
def show_analysis_result():
    user_id = session.get('user_id')  # Ensure you're fetching the user_id
    user_result = result_cache.get(user_id, {})

    # Default to 'Unknown' if no result for skin_type or 'No issues detected' for issues
    detected_skin_type = user_result.get('skin_type', 'Unknown')
    detected_issues = user_result.get('skin_issues', ['No issues detected'])

    return render_template('skin_analysis_result.html', skin_type=detected_skin_type, skin_issues=detected_issues)

@app.route('/about_issues')
def about_issues():
    if 'skin_issues' not in session:
        return redirect(url_for('skin_analysis'))

    issues = session['skin_issues']
    about_info = []

    for issue in issues:
        matches = df_food[
            df_food['skin_issues']
            .str.lower()
            .str.contains(issue.lower())
        ]

        for _, row in matches.iterrows():
            about_info.append({
                'issue': issue,
                'description': row['about']
            })

    return render_template(
        'about_issues.html',
        about_info=about_info
    )


@app.route('/food_recommendations')
def food_recommendations():
    if 'skin_issues' not in session:
        return redirect(url_for('skin_analysis'))

    issues = session['skin_issues']
    food_suggestions = []

    # Loop through the skin issues and match with foods
    for issue in issues:

        matches = df_food[
            df_food['skin_issues'].str.lower() == issue.lower()
        ]

        for _, row in matches.iterrows():
            food_suggestions.append({
                'issue': issue,
                'food': row['food']
            })

    # If no food suggestions, show a default message
    if not food_suggestions:
        food_suggestions.append({
            'issue': "No issues",
            'food': "Remember healthy skin is a journey, not a destination. "
                    "To keep that glow going, nourish yourself with foods "
                    "rich in vitamin C, like citrus fruits, guavas and broccoli. "
                    "Vitamin E rich foods like almonds can also help."
        })

    return render_template(
        'food_recommendations.html',
        food_suggestions=food_suggestions
    )


def hybrid_recommend(
    user_id,
    skin_type,
    skin_issues,
    top_n=5,
    w_cbf=0.6,
    w_cf=0.4
):

    # Combine skin type and issues for content-based filtering
    input_features = vectorizer.transform(
        [skin_type + " " + " ".join(skin_issues)]
    )

    # Calculate cosine similarity
    cbf_scores = cosine_similarity(
        input_features,
        tfidf_matrix
    ).flatten()

    # Collaborative filtering scores
    cf_scores = np.zeros(len(df_products))

    for idx, pid in enumerate(df_products.index):

        try:
            prediction = model.predict(user_id, int(pid))
            cf_scores[idx] = prediction.est

        except:
            cf_scores[idx] = 3.0

    # Combine both scores
    final_scores = (
        w_cbf * cbf_scores
    ) + (
        w_cf * cf_scores
    )

    # Get top recommended products
    top_indices = np.argsort(final_scores)[-top_n:][::-1]

    return df_products.iloc[top_indices][
        ['Product', 'product_url']
    ]


@app.route('/recommend')
def recommend_products():

    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']

    skin_type = session.get(
        'skin_type',
        "Unknown"
    )

    skin_issues = session.get(
        'skin_issues',
        ["General"]
    )

    # Get recommendations
    recommendations = hybrid_recommend(
        user_id,
        skin_type,
        skin_issues
    )

    # Check if ratings were successfully submitted
    rated_successfully = (
        request.args.get('rated') == 'true'
    )

    return render_template(
    'recommend.html',
    recommendations=recommendations,
    rated_successfully=rated_successfully
)





@app.route('/rate_product', methods=['POST'])
def rate_product():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    skin_type = session['skin_type']
    skin_issues = session['skin_issues']

    global df_ratings
    df_ratings = pd.read_csv(ratings_file)

    for product_name in request.form:
        if product_name.startswith("rating_"):
            product_name_clean = product_name.replace("rating_", "").strip()

            rating = request.form[product_name]

            # Normalize both sides to avoid casing/matching issues
            match = df_products[df_products['Product'].str.strip().str.lower() == product_name_clean.lower()]

            if not match.empty:
                product_id = match.index[0]

                new_rating = pd.DataFrame(
                    [[user_id, product_id, skin_type, " ".join(skin_issues), rating]],
                    columns=df_ratings.columns
                )
                df_ratings = pd.concat([df_ratings, new_rating], ignore_index=True)
            else:
                print(f"⚠️ No match found for product: '{product_name_clean}'")

    df_ratings.to_csv(ratings_file, index=False)

    return redirect(url_for('recommend_products'))





if __name__ == '__main__':
    app.run()
