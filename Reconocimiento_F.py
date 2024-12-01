import cv2
import os
from fer import FER

dataPath = r'C:\Users\Sanma\Desktop\Reconocimiento Facial\Data'  # Usar cadena cruda para evitar errores de escape

imagePaths = os.listdir(dataPath)
print('imagePath=', imagePaths)

# Inicializar el detector de emociones
emotion_detector = FER()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar el clasificador de caras preentrenado
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()  # Corregido: .copy() para hacer una copia adecuada
    
    # Detección de rostros
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Detectar emociones en el rostro
        emotion, score = emotion_detector.top_emotion(frame)  # Detección de la emoción principal
        print(f'Emoción detectada: {emotion} con confianza: {score}')
        
        # Mostrar los resultados en la ventana
        if emotion is not None:
            cv2.putText(frame, f'Emocion: {emotion}', (x, y-30), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Emoción no detectada', (x, y-30), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Mostrar el frame con el reconocimiento de emociones
    cv2.imshow('frame', frame)
    
    # Presiona 'Esc' para salir
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
