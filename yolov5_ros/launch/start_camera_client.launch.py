from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    package_name = 'yolov5_ros'
    pkg_dir_yolov5 = os.path.join(get_package_share_directory(package_name ))


    return LaunchDescription([
        #IncludeLaunchDescription(
        #    PythonLaunchDescriptionSource([pkg_dir_ros2_scan_merger  , '/launch/merge_2_scan.launch.py']),
        #),
        TimerAction(
            period=0.5, 
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([pkg_dir_yolov5, '/start_camera.launch.py']),
                ),
            ]
        ),
        TimerAction(
            period=5.5, 
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([pkg_dir_yolov5, '/client.launch.py']),
                ),
            ]
        ),
    ])