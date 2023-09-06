import launch
import launch_ros.actions
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from launch.actions import IncludeLaunchDescription, TimerAction

def generate_launch_description():
    yolox_ros_share_dir = get_package_share_directory('yolov5_ros')

    #webcam = launch_ros.actions.Node(
    #    package="v4l2_camera", executable="v4l2_camera_node",
    #    parameters=[
    #        {"image_size": [640,480]},
    #    ],
    #)

    yolov5_ros = launch_ros.actions.Node(
        package="yolov5_ros", executable="yolov5_ros",
        parameters=[
            {"view_img":True},
        ],

    )

    rqt_graph = launch_ros.actions.Node(
        package="rqt_graph", executable="rqt_graph",
    )

    package_name = 'yolov5_ros'
    rviz_file = os.path.join(get_package_share_directory(package_name ),'test.rviz')
    #pkg_dir_realsense2_camera   = get_package_share_directory('realsense2_camera')

    dir = os.path.join(get_package_share_directory(package_name ),'start_camera.launch.py')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        name='rviz_node',
        parameters=[{'use_sim_time': True}],
        arguments=['-d', rviz_file]
    )

    return launch.LaunchDescription([
        #webcam,
        rviz_node,
        yolov5_ros,
    ])