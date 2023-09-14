import cv2 as cv

capturarVideo = cv.VideoCapture(0)

if not capturarVideo.isOpened():
    print("No se encontro ninguna camara")
    exit()
while True:
    _,camara = capturarVideo.read()
    grises = cv.cvtColor(camara,cv.COLOR_BGR2GRAY)
    cv.imshow("En Directo", camara)
    if cv.waitKey(1) == ord("q"):  # se pone 1 porque es un video y son imagenes continuas
        break
capturarVideo.release()
cv.destroyAllWindows()                