from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load model
model = load_model("model/lstm_mobilenet_skeleton.h5")

# Class labels
classes = [
    'bye', 'computer', 'good', 'hello', 'help', 'hi', 'is', 'like',
    'my', 'name', 'problem', 'see', 'student', 'teacher', 'thank', 'what'
]

# Sentence memory buffer
sentence_buffer = []

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7

# Log file path
LOG_FILE = "logs/sentence_log.txt"

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    global sentence_buffer
    try:
        # ✅ Step 1: Read JSON data from frontend
        data = request.get_json()

        # ✅ Step 2: Convert frames to numpy array with correct shape
        input_array = np.array(data["frames"]).reshape((1, 30, 224, 224, 3))

        # ✅ Step 3: Predict
        prediction = model.predict(input_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence < 0.7:
            return jsonify({
                "message": "Prediction confidence too low",
                "confidence": confidence
            })

        predicted_word = classes[predicted_index]

        # Optional sentence buffer
        sentence_buffer.append(predicted_word)
        if len(sentence_buffer) > 10:
            sentence_buffer = sentence_buffer[-10:]

        return jsonify({
            "word": predicted_word,
            "confidence": confidence,
            "sentence": " ".join(sentence_buffer)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get-sign-video", methods=["POST"])
def get_sign_video():
    data = request.get_json()
    word = data.get("word", "").lower()
    video_path = os.path.join("static/videos", f"{word}.mp4")

    if os.path.exists(video_path):
        return send_from_directory("static/videos", f"{word}.mp4")
    else:
        return jsonify({"message": "Word unavailable"}), 404

@app.route("/clear-sentence", methods=["POST"])
def clear_sentence():
    global sentence_buffer
    sentence_buffer = []
    return jsonify({"message": "Sentence buffer cleared"}), 200

def log_sentence(sentence):
    """Append sentence with timestamp to log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {sentence}\n")

if __name__ == "__main__":
    app.run(debug=True)
