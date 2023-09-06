import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped, Pose


class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Pose, 'pose_topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.pos = Pose()
    
    def timer_callback(self):
        self.pos.position.x = 1.0
        self.pos.position.y = 2.0
        self.pos.position.z = 3.0

        self.publisher_.publish(self.pos)
        self.get_logger().info('Publishing: "%f"' % self.pos.position.x)



def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()