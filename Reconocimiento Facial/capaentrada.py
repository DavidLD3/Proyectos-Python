import cv2 as cv
import os


model = "Fotos David"
ruta1 = "C:/Proyectos Python/Reconocimiento Facial"
rutacompleta = ruta1 + "/" + model
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

camara = cv.VideoCapture(0)

ruidos = cv.CascadeClassifier('C:\Proyectos Python\Reconocimiento Facial\haarcascade_frontalface_default.xml')

id = 0

while True:
    respuesta,captura = camara.read()
    if respuesta == False:
        break

    makegrey = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)

    idcaptura = captura.copy()

    checkface = ruidos.detectMultiScale(makegrey,1.3,5)

    for(x,y,e1,e2) in checkface:
        cv.rectangle(captura,(x,y), (x+e1,y+e2),(0,255,0),2)
        rostrocapturado = idcaptura[y:y+e2,x:x+e1]
        rostrocapturado = cv.resize(rostrocapturado, (170,170), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta + '/imagen_{}.jpg'.format(id), rostrocapturado)
        id = id+1

    cv.imshow("Result face",captura)

    if id == 351:
        break

camara.release()
cv.destroyAllWindows()