#!/usr/bin/env python3
"""
Obtiene el valor del qr
"""
from PIL import Image
from pyzbar.pyzbar import decode

image = Image.open('/home/blank/driver_ws/src/YOLOv5-ROS/yolov5_ros/yolov5_ros/qr.png')
qr_code = decode(image)[0]
# Convert into string
data = qr_code.data.decode('utf8')
#data = qr_code.data.decode('utf8').encode('shift-jis').decode('utf-8')

print("el mensaje es:", data)

# anaqueles = {
#     "A1-N1-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A1-N1-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A1-N2-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A1-N2-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A2-N1-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A2-N1-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A2-N2-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A2-N2-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A3-N1-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A3-N1-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A3-N2-P1": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     },
#     "A3-N2-P2": {
#         "robot_pose": {"X": None, "Y": None, "theta": None},
#         "motor_position": None
#     }
# }

# anaqueles["A1-N1-P1"]["robot_pose"]["X"] = 10.5
