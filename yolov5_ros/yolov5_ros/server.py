#!/usr/bin/env python3
from custom_interfaces.srv import Tf

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Pose

class TfService(Node):

    def __init__(self):
        super().__init__('Tf_server_node')
        self.srv = self.create_service(Tf, 'get_tf_server', self.server_callback)
        #self.broadcaster_t1 = tf2_ros.TransformBroadcaster(self)
        #self.broadcaster_t2 = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(0.1, self.publish_tf)
        self.t1 = TransformStamped()
        self.t2 = TransformStamped()
        self.pub_tf = False
        self.get_logger().info('Service Activate')

    def publish_tf(self):

        if self.pub_tf:
            #self.broadcaster.sendTransform(self.t1)
            #self.broadcaster.sendTransform(self.t2)
            self.get_logger().info('TF')
            pass

    def server_callback(self, request, response):

        self.get_logger().info('Works')

        # self.t1 = TransformStamped()
        # self.t1.header.stamp = self.get_clock().now().to_msg()
        # self.t1.header.frame_id = "camera_link"
        # self.t1.child_frame_id = "objeto_link"

        # # Asignar la posición (aquí se asume que ya tienes las coordenadas X y Y)
        # self.t1.transform.translation.x = request.x /1000
        # self.t1.transform.translation.y = request.y /1000
        # self.t1.transform.translation.z = request.z  /1000 # Ajusta si es necesario

        # # Asumiendo que tienes quaternion (qx, qy, qz, qw) para la rotación de la cámara
        # self.t1.transform.rotation.x = 0.0
        # self.t1.transform.rotation.y = 0.0
        # self.t1.transform.rotation.z = 0.0
        # self.t1.transform.rotation.w = 1.0

        # # Publicar la transformación
        # self.broadcaster.sendTransform(self.t1)

        # #publicar transformacion
        # self.t2.header.stamp = self.get_clock().now().to_msg()
        # self.t2.header.frame_id = "objeto_link"
        # self.t2.child_frame_id = "objeto_goal"

        # # Asignar la posición (aquí se asume que ya tienes las coordenadas X y Y)
        # self.t2.transform.translation.x = request.x /1000 + 0.07; 
        # self.t2.transform.translation.y = request.y /1000
        # self.t2.transform.translation.z = request.z  /1000 # Ajusta si es necesario

        # # Asumiendo que tienes quaternion (qx, qy, qz, qw) para la rotación de la cámara
        # self.t2.transform.rotation.x = 0.0
        # self.t2.transform.rotation.y = 0.0
        # self.t2.transform.rotation.z = 0.0
        # self.t2.transform.rotation.w = 1.0

        # # Publicar la transformación
        # self.broadcaster.sendTransform(self.t2)
        self.pub_tf = True
        response.succeed = True
        return response

def main():
    rclpy.init()
    service = TfService()
    rclpy.spin(service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()