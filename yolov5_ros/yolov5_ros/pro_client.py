#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from custom_interfaces.srv import Tf
from pyzbar.pyzbar import decode
from PIL import Image as PILImage
from std_msgs.msg import Bool
import time

class ImageProcessor(Node):

    def __init__(self):
        super().__init__('image_processor')

        # Suscripciones
        self.image_subscription = self.create_subscription(Image, 
                                                           '/camera/color/image_raw',
                                                           self.image_callback, 10)
        self.depth_subscription = self.create_subscription(Image, 
                                                            '/camera/aligned_depth_to_color/image_raw',
                                                            self.depth_callback, 1)

   
        # self.depth_subscription = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 1)

        self.image_publisher = self.create_publisher(Image, '/processed_image', 3)
        self.image_publisher_depth = self.create_publisher(Image, '/processed_image_depth', 3)



        
        self.image_publisher_depth = self.create_publisher(Image, '/processed_image_depth', 3)

        self.bridge = CvBridge()
        #self.lower_celeste = np.array([80, 180, 130])   # Valor mínimo de tono, saturación y brillo
        #self.upper_celeste = np.array([120, 255, 180])   # Valor maximo de tono, saturación y brillo

        self.lower_celeste = np.array([80, 200, 80])   # Valor mínimo de tono, saturación y brillo
        self.upper_celeste = np.array([140, 255, 120])   # Valor maximo de tono, saturación y brillo

        #self.lower_celeste = np.array([50, 100, 50])   # Valor mínimo de tono, saturación y brillo
        #self.upper_celeste = np.array([120, 200, 100])   # Valor maximo de tono, saturación y brillo


        self.X=0
        self.Y=0
        self.Z=0
        self.str_qr = ''

        #Inicializo centroides 
        self.cX = 0
        self.cY = 0

        #Variables de depth_callback
        self.depth_image = None
        self.last_distance=0
        self.distance = 0
        self.start_depth_callback = False

        self.activar_servicio = False
        #Cliente
        self.cli = self.create_client(Tf, 'get_tf_server')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Tf.Request()

        self.timer = self.create_timer(0.1, self.send_params)
        self.activate_timer = False
        #Inicia todo 
        #self.start()
        self.found_circle = False
        self.get_logger().info('Client Activate')

    # def start(self):
    #     self.image_subscription = self.create_subscription(Image, 
    #                                                        '/camera/color/image_raw',
    #                                                        self.image_callback, 10)
    #     self.depth_subscription = self.create_subscription(Image, 
    #                                                         '/camera/aligned_depth_to_color/image_raw',
    #                                                         self.depth_callback, 1)
    
    #def img_to_str(self, img):
     #   pil_image = PILImage.fromarray(img)
      #  qr_code = decode(pil_image)[0]
       # data = qr_code.data.decode('utf8') #Str
        #self.str_qr = data
        #self.get_logger().info(f'El mensaje es: {data}')
        # self.get_logger().info(f'El mensaje es: ')

    def img_to_str(self, img):
        pil_image = PILImage.fromarray(img)
        qr_codes = decode(pil_image)
        
        if qr_codes:  # Checks if there are any QR codes found
            qr_code = qr_codes[0]
            data = qr_code.data.decode('utf8')
            self.str_qr = data
            self.get_logger().info(f'El mensaje es: {data}')
        else:
            self.get_logger().info('No QR codes found in the image.')

    def send_params(self):
        #self.get_logger().info('Out of Timer')
        if self.activate_timer:
            x = float(self.X)
            y = float(self.Y)
            z = float(self.Z)
            data_qr = self.str_qr
            response = self.send_request(x,y,z,data_qr)
            self.get_logger().info('Result of Tf_server')
            self.timer.cancel()

    def send_request(self,x,y,z,data_qr):
        for a in range(1):
            self.req.x = x
            self.req.y = y
            self.req.z = z
            self.req.qr_str = data_qr
            self.future = self.cli.call_async(self.req)
            rclpy.spin_until_future_complete(self, self.future)

        return self.future.result()

    def image_callback(self, msg):
        #self.get_logger().info('img_callback')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_image(cv_image)



    def depth_callback(self, msg):

        if self.start_depth_callback:
            self.get_logger().info('start_depth_callback')

            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            if int(self.cX) >640 or int(self.cY) >480:
                self.get_logger().info(f"fuera de rango")
                a=2
            else:
                self.distance = self.depth_image[int(self.cY), int(self.cX)]
                if( self.distance == 0):
                    self.distance = self.last_distance
                else:
                    self.distance = self.depth_image[int(self.cY), int(self.cX)]
                    #self.get_logger().info(f"Distancia del círculo en Deeph ({int(cX)}, {int(cY)}): {self.distance} mm")
                    self.last_distance= self.distance
                    self.pixel_to_3d(self.cX, self.cY, self.distance)

            # Sombrea el círculo en la imagen de profundidad y muestra
            #cv2.circle(self.depth_image, ( int(self.cX), int(self.cY) ), 10, (0, 0, 0), -1)  # Radio de 50, ajusta como necesites
            #Dibuja circulo
            #cv2.imshow('Depth with Circle', self.depth_image)

            # Convertir la imagen en un mensaje Image
            processed_img_msg_depth = self.bridge.cv2_to_imgmsg(self.depth_image, encoding='32FC1')
            # Establecer el encabezado del mensaje con la marca de tiempo y el marco de referencia
            processed_img_msg_depth.header.stamp = self.get_clock().now().to_msg()
            processed_img_msg_depth.header.frame_id = 'processed_image_frame_2'  # Cambia al marco de referencia adecuado
            # Publicar la imagen procesada
            self.image_publisher_depth.publish(processed_img_msg_depth)
            #self.depth_subscription.destroy()

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
        
        #Enviar esto por cliente

        self.activate_timer = True

    def process_image(self, img):
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_celeste, self.upper_celeste)
        #Mostrar mascara
        #cv2.imshow('Depth with Circle', mask)
        # Convertir la imagen en un mensaje Image
        processed_img_msg = self.bridge.cv2_to_imgmsg(mask, encoding='8UC1')
        # Establecer el encabezado del mensaje con la marca de tiempo y el marco de referencia

        #################     Publicacion de imagen procesada     #################

        processed_img_msg.header.stamp = self.get_clock().now().to_msg()
        processed_img_msg.header.frame_id = 'processed_image_frame_1'  # Cambia al marco de referencia adecuado
        self.image_publisher.publish(processed_img_msg)

        #################                                         #################

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(f"Cantidad de contornos: {len(contours)}")
        self.img_to_str(img)

        if len(contours) > 0:
            # Solo si se encontraron contornos
            largest_contour = max(contours, key=cv2.contourArea)

            # Calcular el área del contorno más grande
            largest_area = cv2.contourArea(largest_contour)
            #print(f"Área del contorno más grande: {largest_area}")

            # Procesar el contorno más grande y realizar otras operaciones aquí
            cv2.drawContours(img, [largest_contour], 0, (0,255,0), 2)

            M = cv2.moments(largest_contour)
            self.get_logger().info(f"Close")

            #Si el area es distinta de 0
            if M["m00"] != 0:

                #Calculo los centroides
                self.cX = int(M["m10"] / M["m00"])
                self.cY = int(M["m01"] / M["m00"])

                cv2.circle(img, (self.cX, self.cY), 5, (0, 0, 255), -1)
                #self.get_logger().info(f"Distancia del círculo en RGB({cX}, {cY}")
                self.get_logger().info(f"found Circle")
                #Activo depth_subscription
                self.start_depth_callback = True
                self.found_circle = True
                #Activo depth_subscription
                #self.depth_subscription = self.create_subscription(Image, 
                #                                                '/camera/aligned_depth_to_color/image_raw',
                #                                                self.depth_callback, 1)
                cv2.waitKey(1)

        else:
            self.get_logger().info(f"No se encontraron contornos en la imagen.")

            b=2
        #cv2.imshow('Result', img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    rclpy.spin(image_processor)
    image_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()