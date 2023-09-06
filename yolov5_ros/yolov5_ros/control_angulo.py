import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
 
from tf2_msgs.msg import TFMessage
import math

class AngleController(Node):

    def __init__(self):
        super().__init__('angle_controller')
        self.subscription = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.odom_callback,
            10)
        self.publisher_ = self.create_publisher(Twist, '/diffbot_base_controller/cmd_vel_unstamped', 10)
        self.target_angle = math.pi /4 # 90 degrees in radians
        self.Kp = 0.3  # Constante proporcional
        self.Ki = 0.01  # Constante integral
        self.integral_error = 0  # Error acumulado


    def odom_callback(self, msg):
        # Extract yaw from the odometry quaternion
        _, _, yaw = self.euler_from_quaternion(msg.pose.pose.orientation)
        self.get_logger().info(f"Current Yaw: {yaw * 180 / math.pi} degrees")
        angular_error = self.target_angle - yaw
        self.get_logger().info(f"Error: {angular_error * 180 / math.pi} degrees")
        self.get_logger().info(f"target angle: {self.target_angle * 180 / math.pi} degrees")
        if (abs(angular_error)) >0.1:
            # Accumulate error
            self.integral_error += angular_error

            # PI controller for angle control
            control_output = self.Kp * angular_error + self.Ki * self.integral_error
            if control_output > 0.3:
                control_output = 0.3
            if control_output < -0.3:
                control_output = -0.3
            i
            # Create and publish the control command
            twist = Twist()
            twist.angular.z = control_output
            #self.publisher_.publish(twist)

    def euler_from_quaternion(self, quaternion):
        """
        Convert quaternion to euler roll, pitch, yaw
        quaternion: geometry_msgs/Quaternion
        r, p, y = euler_from_quaternion(quaternion)
        """
        q = [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w
        ]

        sinr_cosp = 2 * (q[3] * q[0] + q[1] * q[2])
        cosr_cosp = 1 - 2 * (q[0] * q[0] + q[1] * q[1])
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q[3] * q[1] - q[2] * q[0])
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2 * (q[3] * q[2] + q[0] * q[1])
        cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
        # Ask user for desired angle
    try:
        angle_deg = float(input("Please enter the desired angle in degrees: "))
        target_angle = math.radians(angle_deg)  # Convert to radians
        print("Numero ingresado.")
    except ValueError:
        print("Invalid input. Please provide a number.")
        return
    rclpy.init(args=args)
    controller = AngleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()