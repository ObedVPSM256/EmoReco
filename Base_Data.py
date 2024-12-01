import cv2
import os
import cv2.data
import imutils

# Nombre de la persona
personName = 'Obed'
# Ruta donde se guardarán los datos
dataPath = r'C:\Users\Sanma\Desktop\Reconocimiento Facial\Data'  # Usar cadena cruda para evitar errores de escape
personPath = os.path.join(dataPath, personName)  # Usar os.path.join para concatenar rutas de manera segura

# Crear la carpeta si no existe
if not os.path.exists(personPath):
    print('Carpeta creada:', personPath)
    os.makedirs(personPath)

# Capturar video desde la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

while True:
    ret, frame = cap.read()  # Leer el frame de la cámara
    if not ret:
        break

    frame = imutils.resize(frame, width=640)  # Redimensionar el frame para mayor rendimiento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    auxFrame = frame.copy()
    
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
        count = count + 1

    # Mostrar el frame en una ventana
    cv2.imshow('Frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 300:
        break
        

# Liberar la captura de video y cerrar las ventanas abiertas
cap.release()
cv2.destroyAllWindows()
