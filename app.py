# ---------------------------------------
# STEP 3: FLASK APPLICATION
# ---------------------------------------

from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Folder where uploaded images go temporarily
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------
# Step 3.1 — Load pretrained model (same as feature_extraction.py)
# ---------------------------------------
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# ---------------------------------------
# Step 3.2 — Load stored features and image paths
# ---------------------------------------
features = np.load("features.npy")
image_paths = np.load("image_paths.npy", allow_pickle=True)

# ---------------------------------------
# Step 3.3 — Function to extract feature vector of uploaded image
# ---------------------------------------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    feature = model.predict(x)[0]
    feature = feature / np.linalg.norm(feature)
    return feature


# ---------------------------------------
# Step 3.4 — Routes
# ---------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():

    # Step 3.4.1 — Get uploaded file
    file = request.files["query_img"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Step 3.4.2 — Extract its features
    query_feature = extract_features(file_path)

    # Step 3.4.3 — Compute cosine similarity with all stored features
    sims = cosine_similarity([query_feature], features)[0]

    # Step 3.4.4 — Get top 5 similar images (highest values)
    top_indices = sims.argsort()[-5:][::-1]

    results = [(image_paths[i], sims[i]) for i in top_indices]

    return render_template("results.html", query_image=file_path, results=results)


if __name__ == "__main__":
    app.run(debug=True)
