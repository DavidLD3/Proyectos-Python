import cv2
image = cv2.imread('C:\Proyectos Python\Monedas Contorno\contorno.jpg')

makegrey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  # Darle color grises a la imagen
_,umbral = cv2.threshold(makegrey,100,255,cv2.THRESH_BINARY) ## al poner _ adopta una variable ficticia y lo ignora, porque threshold devuelve 2 variables
contorno,jerarquia = cv2.findContours(umbral,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) ## Se usa para dibujar los contornos de los bordes del dibujo
cv2.drawContours(image,contorno,-1,(251, 63, 52),3)

#Codigo para Mostrar imagenes
cv2.imshow('Imagen original',image)  #para enseñar la imagen
#cv2.imshow('Imagen en grises', makegrey)
#cv2.imshow('Imagen Umbral', umbral)
cv2.waitKey(0) # al presionar una tecla se ejecuta
cv2.destroyAllWindows()  # Para cerrar todas las ventanas al cerrar 11