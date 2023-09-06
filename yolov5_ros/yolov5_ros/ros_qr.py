import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode
from PIL import Image as PILImage

class QrClass(Node):
    def __init__(self):
        super().__init__('qr_class_node')

        # Sub
        self.image_subscription = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.rgb_image = None
        self.data = ""

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.img_to_str(cv_image)

    def img_to_str(self, img):
        pil_image = PILImage.fromarray(img)
        qr_code = decode(pil_image)[0]
        data = qr_code.data.decode('utf8') #Str
        self.get_logger().info('El mensaje es: %s' % data)

def main(args=None):
    rclpy.init(args=args)
    qr_node = QrClass()
    rclpy.spin(qr_node)
    qr_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()