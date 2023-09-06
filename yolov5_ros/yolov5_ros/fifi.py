import cv2
import numpy as np
import matplotlib.pyplot as plt

################ LAST
import cv2
import numpy as np

# Lee la imagen
img = cv2.imread('/home/blank/driver_ws/src/YOLOv5-ROS/yolov5_ros/foto4.jpeg')
assert img is not None, "Error: Imagen no encontrada o no pudo ser leída."

# 2. Define el rango para el color azul en el espacio HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([85, 100, 100])
upper_blue = np.array([100, 255, 255])

# 3. Crea una máscara con los píxeles que estén dentro del rango azul
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 4. Aplica la máscara a la imagen
img_masked = cv2.bitwise_and(img, img, mask=mask)

# 5. Encuentra los contornos en la máscara
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 6. Dibuja los contornos encontrados en la imagen aplicada a la máscara en la imagen principal
for contour in contours:
    # Dibuja el contorno
    cv2.drawContours(img, [contour], 0, (0,255,0), 2)
    
    # Encuentra el centro del contorno (círculo azul)
    M = cv2.moments(contour)
    if M["m00"] != 0: # Para evitar una división por cero
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    # Dibuja el centro
    cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)  # Dibuja en rojo el centro

# Muestra la imagen resultante
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


