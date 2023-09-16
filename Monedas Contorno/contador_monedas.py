import cv2
import numpy as np

def ordenarpuntos(puntos):
    n_puntos = np.concatenate([puntos[0],puntos[1],puntos[2],puntos[3]]).tolist()
    y_order = sorted(n_puntos,key = lambda n_puntos:n_puntos[1])
    x1_order = y_order[0:2]
    x1_order = sorted(x1_order, key = lambda x1_order:x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key = lambda x2_order:x2_order[0])
    return [x1_order[0],x1_order[1],x2_order[0],x2_order[1]]

def aliniamiento(imagen, ancho, alto):
    imagen_alineada = None
    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, umbral = cv2.threshold(grises, 150, 255,cv2.THRESH_BINARY)
    cv2.imshow("Umbral", umbral)
    contorno = cv2.findContours(umbral, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key = cv2.contourArea,reverse=True)[:1]

    for i in contorno:
        epsilon = 0.01*cv2.arcLength(i, True)
        aproximacion = cv2.approxPolyDP(i,epsilon,True)
        if len(aproximacion) == 4:
            puntos = ordenarpuntos(aproximacion)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0],[ancho,0],[0,alto],[ancho,alto]])
            m = cv2.getPerspectiveTransform(puntos1,puntos2)
            imagen_alineada = cv2.warpPerspective(imagen,m,(ancho,alto))
    return imagen_alineada

captura_video = cv2.VideoCapture(0)

while True:
    tipocamara,camara = captura_video.read()
    if tipocamara == False:
        break
    imagen_A6 = aliniamiento(camara,ancho=720,alto=500)
    if imagen_A6 is not None:
        puntos = []
        imagen_gris = cv2.cvtColor(imagen_A6,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris,(5,5),1)
        _,umbral2 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow("Umbral", umbral2)
        contorno2 = cv2.findContours(umbral2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6,contorno2,-1,(255,0,0),2)
        suma1 = 0.0
        suma2 = 0.0
        for c in contorno2:
            area = cv2.contourArea(c)
            momentos = cv2.moments(c)
            if(momentos["m00"] == 0):
                momentos["m00"] = 1.0
            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            if area < 9600 and area > 8000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "0.50 €",(x,y),font,0.75,(0,255,0),2)
                suma1 = suma1+0.5

            if area < 6600 and area > 5500:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "0.10 €",(x,y),font,0.75,(0,255,0),2)
                suma2 = suma2+0.1
        total = suma1 + suma2
        print("Sumatoria total en centimos: ", round(total,2))
        cv2.imshow("Imagen A6", imagen_A6)
        #cv2.inshow("Camara", camara)

    if cv2.waitKey(1) == ord('q'):
        break

captura_video.release()
cv2.destroyAllWindows()   