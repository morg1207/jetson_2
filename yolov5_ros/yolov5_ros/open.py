import cv2

# Cargar la imagen desde un archivo en el sistema de archivos
image_path = '1.png'  # Cambia esto a la ruta de tu imagen
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Verificar si la imagen se cargó correctamente
if image is not None:
    cv2.imshow('Imagen', hsv)  # Mostrar la imagen en una ventana
    
    cv2.waitKey(0)  # Esperar hasta que se presione una tecla
    cv2.destroyAllWindows()  # Cerrar todas las ventanas
else:
    print('No se pudo cargar la imagen.')