import cv2 as cv
import os
import numpy as np
from time import time

dataRuta = "C:\Proyectos Python\Reconocimiento Facial\Data"

listaData = os.listdir(dataRuta)

ids = []
rostrosData = []
id = 0
tiempoinicial = time()

for fila in listaData:
    rutacompleta = dataRuta + '/' + fila
    print("Iniciando lectura")
    for archivo in os.listdir(rutacompleta):
        print("Imagenes: ",fila + "/" + archivo)
        ids.append(id)
        rostrosData.append(cv.imread(rutacompleta + '/' + archivo, 0)) 

    id = id + 1

    tiempofinalLectura = time()
    tiempoTotalLectura = tiempofinalLectura - tiempoinicial 
    print("Tiempo total lectura: " ,tiempoTotalLectura)

entrenamiendoModelo1 = cv.face.EigenFaceRecognizer_create()
print("iniciando el entrenamiento....Espere")
entrenamiendoModelo1.train(rostrosData,np.array(ids))
tiempofinalentrenamiento = time()
tiempoTotalEntrenamiento = tiempofinalentrenamiento - tiempoTotalLectura
print("Tiempo de entrenamiento total: " + tiempoTotalEntrenamiento)
entrenamiendoModelo1.write("entrenamientoEigen.xml")
print("Entrenamiento Concluido")