import cv2 as cv
import os

dataRuta = "C:\Proyectos Python\Reconocimiento Facial\Data"

listaData = os.listdir(dataRuta)

entrenamiendoModelo1 = cv.face.EigenFaceRecognizer_create()
entrenamiendoModelo1.read('EntrenamientoEigenFaceRecognizer.xml')
ruidos=cv.CascadeClassifier('C:\Proyectos Python\Reconocimiento Facial\haarcascade_frontalface_default.xml')
camara=cv.VideoCapture('desco.mp4')
while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
        resultado=entrenamiendoModelo1.predict(rostrocapturado)
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1,1.3,(0,255,0),1,cv.LINE_AA)
        if resultado[1]<8000:
            cv.putText(captura, '{}'.format(listaData[resultado[0]]), (x,y-20), 2,1.1,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)
        else:
            cv.putText(captura,"No encontrado", (x,y-20), 2,0.7,(0,255,0),1,cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)

       
    cv.imshow("Resultados", captura)
    if cv.waitKey(1)==ord('s'):
        break
camara.release()
cv.destroyAllWindows()