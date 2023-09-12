import cv2
import numpy as np

valor_gauss = 3
valor_kernel = 3

image_original = cv2.imread('C:\Proyectos Python\Monedas Contorno\monedas.jpg')

makegrey = cv2.cvtColor(image_original,cv2.COLOR_BGR2GRAY)

desenfoque = cv2.GaussianBlur(makegrey,(valor_gauss,valor_gauss),0)

bordes_canny = cv2.Canny(desenfoque, 60,100)

kernel = np.ones((valor_kernel,valor_kernel),np.uint8) # Cuando son matrices trabajar con 8 bits

cierre_contornos = cv2.morphologyEx(bordes_canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquia = cv2.findContours(cierre_contornos.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )  #el copy copia todos los datos intentos de la variable

print("monedas encontradas : {}".format(len(contornos)))

cv2.drawContours(image_original, contornos, -1, (0,0,255),2)

# Mostar resultados imagenes

cv2.imshow("Imagen en gris",makegrey)
cv2.imshow("Desenfoque",desenfoque)
cv2.imshow("Canny",bordes_canny)
cv2.imshow("Resultado", image_original)
cv2.waitKey(0)

