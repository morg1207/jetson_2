from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov5_ros',
            node_executable='pro_client',
            node_name='server_node',
            output='screen'
        ),
    ])
