from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask_cors import CORS




app = Flask(__name__)
CORS(app)

# Load your trained model
model = load_model("cnn_model.h5")

# Class names (update with actual class names from your dataset)
CLASS_NAMES = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']

# Preprocess function
def preprocess_image(image, target_size=(176, 176)):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0  # Normalize
    return np.expand_dims(image, axis=0)  # Add batch dimension

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = preprocess_image(file)
        prediction = model.predict(image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
