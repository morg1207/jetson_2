import numpy as np
import cv2 as cv

img = cv.imread('/home/blank/driver_ws/src/YOLOv5-ROS/yolov5_ros/foto2.png')
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

cimg = np.max(cimg,axis=2)

sobel_x = cv.Sobel(cimg, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(cimg, cv.CV_64F, 0, 1, ksize=3)
edges_sobel = cv.magnitude(sobel_x, sobel_y)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
 param1=50,param2=30,minRadius=10,maxRadius=70)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
 # draw the outer circle
 cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
 # draw the center of the circle
 cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()