from flask import Flask, request, jsonify, render_template
import cv2
import os
from fer import FER
from flask_cors import CORS
import base64
from io import BytesIO
import numpy as np
from PIL import Image

app = Flask(__name__)
CORS(app)  # Esto habilita CORS

# Inicializar el detector de emociones
emotion_detector = FER()

# Ruta para manejar el reconocimiento facial
@app.route('/Reconocimiento_F', methods=['POST'])
def reconocimiento_facial():
    if request.method == 'POST':
        try:
            data = request.get_json()
            image_data = data.get("image_data")
        
            if not image_data:
                return jsonify({"error": "No se recibió imagen"}), 400
            
            # Decodificar la imagen base64
            image_data = image_data.split(",")[1]  # Eliminar el prefijo "data:image/jpeg;base64,"
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))

            # Convertir a imagen de OpenCV para el análisis
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Detectar la emoción en la imagen
            emotion, score = emotion_detector.top_emotion(img)
        
            if emotion:
                return jsonify({"emotion": emotion, "score": score})
            else:
                return jsonify({"error": "No se pudo detectar ninguna emoción"}), 400

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Ruta de prueba para verificar que el servidor está activo
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Servidor activo"}), 200

# Ruta para servir el frontend HTML
@app.route('/')
def index():
    return render_template('index.html')  # Asegúrate de que el archivo HTML esté en la carpeta 'templates'

if __name__ == "__main__":
    app.run(debug=True)
