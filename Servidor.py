from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from fer import FER
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)

emotion_detector = FER()

@app.route('/Reconocimiento_F', methods=['POST'])
def reconocimiento_facial():
    try:
        data = request.get_json()
        image_data = data.get("image_data")

        if not image_data:
            return jsonify({"error": "No se recibió imagen"}), 400

        # Decodificar la imagen base64
        image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_bytes))

        # Convertir a array de numpy y luego a formato BGR para OpenCV
        img_np = np.array(img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Detectar rostros
        gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        emotions_detected = []
        for (x, y, w, h) in faces:
            face = img_cv2[y:y+h, x:x+w]
            emotions = emotion_detector.detect_emotions(face)
            if emotions:
                emotion, score = max(emotions[0]['emotions'].items(), key=lambda x: x[1])
                cv2.rectangle(img_cv2, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{emotion}: {score:.2f}"
                cv2.putText(img_cv2, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emotions_detected.append({"emotion": emotion, "score": score})

        # Convertir la imagen procesada de vuelta a base64
        _, buffer = cv2.imencode('.jpg', img_cv2)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "processed_image": f"data:image/jpeg;base64,{img_base64}",
            "emotions": emotions_detected
        })

    except Exception as e:
        print(f"Error en el servidor: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
