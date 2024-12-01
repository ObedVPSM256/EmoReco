from flask import Flask, request, jsonify, render_template
import cv2
import os
from fer import FER
from flask_cors import CORS

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
            image_path = data.get("image_path")
        
            if not image_path or not os.path.exists(image_path):
                return jsonify({"error": "Imagen no encontrada o ruta inválida"}), 400
        
            # Leer la imagen
            image = cv2.imread(image_path)
            if image is None:
                return jsonify({"error": "No se pudo leer la imagen"}), 400

            # Detectar la emoción en la imagen
            emotion, score = emotion_detector.top_emotion(image)
        
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
