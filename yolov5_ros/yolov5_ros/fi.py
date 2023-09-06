import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('/home/blank/driver_ws/src/YOLOv5-ROS/yolov5_ros/foto2.png')
image = cv2.medianBlur(image,5)
cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

image = np.max(image, axis=2)
edges_canny = cv2.Canny(image, 100, 200)

sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)

kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
perwitt_x = cv2.filter2D(image, cv2.CV_64F, kx)
perwitt_y = cv2.filter2D(image, cv2.CV_64F, ky)
edges_perwitt = cv2.magnitude(perwitt_x, perwitt_y)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(edges_canny, cmap='gray')
axes[1].set_title('Canny Edges')
axes[1].axis('off')

axes[2].imshow(edges_sobel, cmap='gray')
axes[2].set_title('Sobel Edges')
axes[2].axis('off')

axes[3].imshow(edges_perwitt, cmap='gray')
axes[3].set_title('Perwitt Edges')
axes[3].axis('off')

plt.tight_layout()
plt.show()