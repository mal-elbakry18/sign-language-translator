from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
CORS(app)

model = load_model("model/lstm_mobilenet_skeleton.h5")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json['landmarks']
        input_array = np.array(data).reshape((1, 30, 42))  # Match model input shape
        prediction = model.predict(input_array)
        label = np.argmax(prediction)
        return jsonify({"label": str(label)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory("static/videos", filename)

if __name__ == "__main__":
    app.run(debug=True)
