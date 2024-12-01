import cv2
import os
import numpy as np

dataPath = r'C:\Users\Sanma\Desktop\Reconocimiento Facial\Data'  # Ruta de datos
peopleList = os.listdir(dataPath)
print('Lista de Personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)
    print('Leyendo Imágenes')
    
    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/ ' + fileName)
        
        # Leer la imagen en escala de grises
        imagePath = os.path.join(personPath, fileName)
        image = cv2.imread(imagePath, 0)
        
        if image is None:
            print(f'Error al cargar la imagen: {imagePath}')
            continue  # Saltar esta imagen si hay un error

        labels.append(label)
        facesData.append(image)
        
    label += 1

# Imprimir etiquetas para verificar
print('labels = ', labels)
print('Numero de etiquetas 0: ', np.count_nonzero(np.array(labels) == 0))
print('Numero de etiquetas 1: ', np.count_nonzero(np.array(labels) == 1))

# Crear el reconocedor y entrenar el modelo
face_recognizer = cv2.face.LBPHFaceRecognizer_create()  # Cambiado a LBPH

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Guardar el modelo entrenado
face_recognizer.write('ModeloFaceFrontalData2024.xml')
print("Modelo Guardado")