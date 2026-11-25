# ============================================================
#   Image Caption Generator Web App using Flask
# ============================================================
# This Flask application allows the user to upload an image.
# The uploaded image is processed using OpenCV and VGG16 CNN.
# The trained LSTM model then generates a caption for the image.
# ============================================================


# ============================================================
#   Import Required Libraries
# ============================================================

import os
import cv2
import pickle
import numpy as np

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
#   Create Flask App
# ============================================================
# Flask is used to create the web application.
# The user can upload an image from the browser.

app = Flask(__name__)


# ============================================================
#   Set Upload Folder and Allowed File Types
# ============================================================
# Uploaded images will be saved inside static/uploads.
# Only JPG, JPEG, and PNG files are allowed.

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ============================================================
#   Load Trained Caption Model
# ============================================================
# This is the trained CNN + RNN/LSTM model saved from Colab.
# compile=False is used because we only need prediction, not training.

model = load_model("best_image_caption_model.keras", compile=False)

# ============================================================
#   Load Tokenizer
# ============================================================
# The tokenizer converts words into numbers.
# It must be the same tokenizer used during training.

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# ============================================================
#   Load Maximum Caption Length
# ============================================================
# max_length is used to pad caption sequences to the same length.

with open("max_length.pkl", "rb") as f:
    max_length = pickle.load(f)

# ============================================================
#   Create Index-to-Word Dictionary
# ============================================================
# This dictionary helps convert predicted word IDs back to words quickly.
# It is used during Beam Search caption generation.

vocab_size = model.output_shape[-1]

index_word = {}

for word, index in tokenizer.word_index.items():
    if index < vocab_size:
        index_word[index] = word
        
        
# ============================================================
#   Load VGG16 CNN Model for Feature Extraction
# ============================================================
# VGG16 is a pre-trained CNN model.
# The final classification layer is removed.
# The second-last layer is used to extract image features.

vgg_model = VGG16(weights="imagenet")
vgg_model = Model(
    inputs=vgg_model.inputs,
    outputs=vgg_model.layers[-2].output
)


# ============================================================
#   Function to Check Allowed File Types
# ============================================================
# This function checks whether the uploaded file is an image
# with allowed extensions: png, jpg, or jpeg.

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
#   Function to Extract Image Features
# ============================================================
# OpenCV reads the uploaded image.
# The image is converted from BGR to RGB.
# It is resized to 224x224 pixels because VGG16 requires this size.
# The CNN then extracts a feature vector from the image.

def extract_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not read image. Please upload a valid JPG or PNG file.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)

    feature = vgg_model.predict(image, verbose=0)

    return feature


# ============================================================
#   Function to Generate Caption using Beam Search
# ============================================================
# Beam Search checks multiple possible caption sequences.
# It gives better captions than simple argmax prediction.
# beam_index=5 means the model keeps the top 5 possible caption paths.

def generate_caption(photo, beam_index=5):
    start = "startseq"
    sequences = [(start, 0.0)]

    for _ in range(max_length):
        all_candidates = []

        for in_text, score in sequences:

            # If caption has already ended, keep it as it is
            if in_text.split()[-1] == "endseq":
                all_candidates.append((in_text, score))
                continue

            # Convert text into sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]

            # Same post-padding used during training
            sequence = pad_sequences(
                [sequence],
                maxlen=max_length,
                padding="post"
            )

            # Predict next word probabilities
            yhat = model.predict([photo, sequence], verbose=0)[0]

            # Get top beam_index word IDs
            top_words = np.argsort(yhat)[-beam_index:]

            for word_id in top_words:
                word = index_word.get(word_id)

                if word is None:
                    continue

                if word in ["startseq", "unk"]:
                    continue

                new_text = in_text + " " + word
                new_score = score - np.log(yhat[word_id] + 1e-9)

                all_candidates.append((new_text, new_score))

        # Sort by score and keep best beam_index captions
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:beam_index]

    final_caption = sequences[0][0]

    final_caption = final_caption.replace("startseq", "")
    final_caption = final_caption.replace("endseq", "")
    final_caption = final_caption.strip()

    return final_caption

# ============================================================
#   Main Route for Home Page
# ============================================================
# GET request shows the upload page.
# POST request handles image upload and caption generation.

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_path = None
    error = None

    if request.method == "POST":

        # Check if image file is present in request
        if "image" not in request.files:
            error = "No image selected."
            return render_template(
                "index.html",
                caption=caption,
                image_path=image_path,
                error=error
            )

        file = request.files["image"]

        # Check if user selected a file
        if file.filename == "":
            error = "Please select an image."
            return render_template(
                "index.html",
                caption=caption,
                image_path=image_path,
                error=error
            )

        # Check file extension
        if file and allowed_file(file.filename):

            # Secure the file name before saving
            filename = secure_filename(file.filename)

            # Create full save path
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Save uploaded image
            file.save(save_path)

            try:
                # Extract CNN features from uploaded image
                photo = extract_features(save_path)

                # Generate caption using trained LSTM model
                caption = generate_caption(photo)

                # Convert path for browser display
                image_path = save_path.replace("\\", "/")

            except Exception as e:
                error = str(e)

        else:
            error = "Only JPG, JPEG, and PNG files are allowed."

    return render_template(
        "index.html",
        caption=caption,
        image_path=image_path,
        error=error
    )


# ============================================================
#   Run Flask Application
# ============================================================
# This creates the upload folder if it does not exist.
# The app runs locally on http://127.0.0.1:5000

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)