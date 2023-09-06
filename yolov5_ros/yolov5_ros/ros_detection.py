import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
	
        # Suscripciones
        self.image_subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 1)
        self.depth_subscription = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)


        #self.depth_subscription = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        
        self.image_publisher = self.create_publisher(Image, '/processed_image', 5)
        self.image_publisher_depth = self.create_publisher(Image, '/processed_image_depth', 5)
        self.pub_tf_ = self.create_publisher(Pose, '/circle', 5)

        self.bridge = CvBridge()
        self.depth_image = None
        self.last_distance=0
        self.distance = 0

        self.X=0
        self.Y=0
        self.Z=0
        self.tf_pose = Pose()




    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_image(cv_image)

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

    def pixel_to_3d(self, coord_x, coord_y, depth, image_width=640, image_height=480, fov_h=69, fov_v=42):
        # Convertir FOV a radianes
        fov_h_rad = np.deg2rad(fov_h)
        fov_v_rad = np.deg2rad(fov_v)
        
        # Calcular el tamaño del píxel en radianes
        angle_per_pixel_x = fov_h_rad / image_width
        angle_per_pixel_y = fov_v_rad / image_height
        
        # Calcular el ángulo para el píxel x, y
        theta_x = (coord_x - image_width / 2) * angle_per_pixel_x
        theta_y = (coord_y- image_height / 2) * angle_per_pixel_y
        
        # Calcular las coordenadas X, Y en el plano de la imagen
        self.Y = -depth * np.tan(theta_x)
        self.Z = -depth * np.tan(theta_y)
        self.X = depth
        print(f"Posición 3D del objeto: X = {self.X} m, Y = {self.Y} m, Z = {self.Z} m")
        
        #TF Pose
        self.tf_pose.position.x = float(self.X)
        self.tf_pose.position.y = float(self.Y)
        self.tf_pose.position.z = float(self.Z)

    def process_image(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_celeste = np.array([80, 180, 160])   # Valor mínimo de tono, saturación y brillo
        upper_celeste = np.array([255, 255, 250])   # Valor mínimo de tono, saturación y brillo

        
        mask = cv2.inRange(hsv, lower_celeste, upper_celeste)
        #Para pruebas
        #cv2.imshow('Depth with Circle', mask)
        # Convertir la imagen en un mensaje Image
        processed_img_msg = self.bridge.cv2_to_imgmsg(mask, encoding='8UC1')
        # Establecer el encabezado del mensaje con la marca de tiempo y el marco de referencia
        processed_img_msg.header.stamp = self.get_clock().now().to_msg()
        processed_img_msg.header.frame_id = 'processed_image_frame_1'  # Cambia al marco de referencia adecuado

        # Publicar la imagen procesada
        self.image_publisher.publish(processed_img_msg)

                
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(f"Cantidad de contornos: {len(contours)}")


        if len(contours) > 0:
            # Solo si se encontraron contornos
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calcular el área del contorno más grande
            largest_area = cv2.contourArea(largest_contour)
            #print(f"Área del contorno más grande: {largest_area}")
            
            # Procesar el contorno más grande y realizar otras operaciones aquí
            cv2.drawContours(img, [largest_contour], 0, (0,255,0), 2)
            
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)
                #self.get_logger().info(f"Distancia del círculo en RGB({cX}, {cY}")
                # Obtener distancia del círculo si la imagen de profundidad está disponible
                if self.depth_image is not None:
                    if int(cX) >1280 or int(cY) >720:
                        #self.get_logger().info(f"fuera de rango")
                        a=2
                        
                    else:
                        self.distance = self.depth_image[int(cY), int(cX)]
                        if( self.distance == 0):
                            self.distance = self.last_distance
                        else:
                            self.distance = self.depth_image[int(cY), int(cX)]
                            self.get_logger().info(f"Distancia del círculo en Deeph ({int(cX)}, {int(cY)}): {self.distance} mm")
                            self.pub_tf_.publish(self.tf_pose)

                            self.last_distance= self.distance
                            self.pixel_to_3d(cX, cY, self.distance)
                    # Sombrea el círculo en la imagen de profundidad y muestra
                    cv2.circle(self.depth_image, ( int(cX), int(cY) ), 10, (0, 0, 0), -1)  # Radio de 50, ajusta como necesites
                    #cv2.imshow('Depth with Circle', self.depth_image)
                    # Convertir la imagen en un mensaje Image
                    processed_img_msg_depth = self.bridge.cv2_to_imgmsg(self.depth_image, encoding='32FC1')
                    # Establecer el encabezado del mensaje con la marca de tiempo y el marco de referencia
                    processed_img_msg_depth.header.stamp = self.get_clock().now().to_msg()
                    processed_img_msg_depth.header.frame_id = 'processed_image_frame_2'  # Cambia al marco de referencia adecuado
                    # Publicar la imagen procesada
                    self.image_publisher_depth.publish(processed_img_msg_depth)

                #cv2.waitKey(1)
        else:
            print("No se encontraron contornos en la imagen.")
            b=2

        #cv2.imshow('Result', img)
        #cv2.waitKey(1)
        

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
