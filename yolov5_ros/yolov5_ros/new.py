import argparse
import os
import sys
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import time

import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from custom_interfaces.msg import Circle

from yolov5_ros.models.common import DetectMultiBackend
from yolov5_ros.utils.datasets import IMG_FORMATS, VID_FORMATS
from yolov5_ros.utils.general import (LOGGER, check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh)
from yolov5_ros.utils.plots import Annotator, colors
from yolov5_ros.utils.torch_utils import select_device, time_sync

from yolov5_ros.utils.datasets import letterbox

from rclpy.qos import QoSProfile, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox
from std_msgs.msg import Header
from cv_bridge import CvBridge


#Definimos QoS para mensajes fijos
latched_qos = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL
)

class yolov5_demo():
    def __init__(self,  weights,
                        data,
                        imagez_height,
                        imagez_width,
                        conf_thres,
                        iou_thres,
                        max_det,
                        device,
                        view_img,
                        classes,
                        agnostic_nms,
                        line_thickness,
                        half,
                        dnn
                        ):
        self.weights = weights
        self.data = data
        self.imagez_height = imagez_height
        self.imagez_width = imagez_width
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.line_thickness = line_thickness
        self.half = half
        self.dnn = dnn

        self.s = str()

        self.load_model()

    def load_model(self):
        imgsz = (self.imagez_height, self.imagez_width)

        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data)
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        source = 0
        # Dataloader
        webcam = True
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
        bs = 1
        self.vid_path, self.vid_writer = [None] * bs, [None] * bs

        self.model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0], 0

    # callback ==========================================================================

    # return ---------------------------------------
    # 1. class (str)                                +
    # 2. confidence (float)                         +
    # 3. x_min, y_min, x_max, y_max (float)         +
    # ----------------------------------------------
    def image_callback(self, image_raw):
        class_list = []
        confidence_list = []
        x_min_list = []
        y_min_list = []
        x_max_list = []
        y_max_list = []

        # im is  NDArray[_SCT@ascontiguousarray
        # im = im.transpose(2, 0, 1)
        self.stride = 32  # stride
        self.img_size = 640
        img = letterbox(image_raw, self.img_size, stride=self.stride)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(img)

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        save_dir = "runs/detect/exp7"
        path = ['0']

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        self.dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):
            im0 = image_raw
            self.s += f'{i}: '

            # p = Path(str(p))  # to Path
            self.s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    self.s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    save_conf = False
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    
                    # Add bbox to image
                    c = int(cls)  # integer class
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # print(xyxy, label)
                    class_list.append(self.names[c])
                    confidence_list.append(conf)
                    # tensor to float
                    x_min_list.append(xyxy[0].item())
                    y_min_list.append(xyxy[1].item())
                    x_max_list.append(xyxy[2].item())
                    y_max_list.append(xyxy[3].item())

            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow("yolov5", im0)
                cv2.waitKey(1)  # 1 millisecond

            return class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list




class yolov5_ros(Node):
    def __init__(self):

        super().__init__('yolov5_ros')

        self.my_callback_group = ReentrantCallbackGroup()

        
        self.sub_image_ = self.create_subscription(Image, '/camera/color/image_raw', 
                                                   self.image_callback,10,
                                                   callback_group=self.my_callback_group)
        
        self.sub_image_depth_ = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', 
                                                         self.aligned_depth_to_color_callback,10,
                                                         callback_group=self.my_callback_group)

        
        #CV 
        self.bridge = CvBridge()
        self.start_img_callback = True

        #Publisher_
        self.pub_bbox_ = self.create_publisher(BoundingBoxes, 'yolov5/bounding_boxes', 10)
        self.pub_image_ = self.create_publisher(Image, 'yolov5/image_raw',  qos_profile=latched_qos)
        self.pub_image_with_circle_pub_ = self.create_publisher(Image, 'yolov5/final_img',  qos_profile=latched_qos)
        self.pub_subimage_ = self.create_publisher(Image, 'yolov5/sub_image',  qos_profile=latched_qos)
        self.circle_pub_ = self.create_publisher(Circle, 'yolov5/circle_info', qos_profile=latched_qos)
        #self.circle_sub_ = self.create_subscription(Circle, '/yolov5/circle_info', self.circle_sub_callback,qos_profile=latched_qos)

        #Subscriber_
        self.sub_image_ = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback,10)
        self.sub_image_depth_ = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.aligned_depth_to_color_callback,10)

        #Pkg name
        self.package_path = get_package_share_directory('yolov5_ros')

        #Timer
        #self.timer_period = 0.5
        #self.timer = self.create_timer(self.timer_period, self.timer_callback)

        #Inference params
        self.data_file = os.path.join(self.package_path,'yolov5_ros' ,'data', 'data_2.yaml')
        self.weights_file = os.path.join(self.package_path,'yolov5_ros' ,'config', 'best_2.pt')
        
        #Img variables
        self.last_image = None
        self.last_result = None
        self.msg = None
        self.img_pub = None
        self.image_with_bboxes = None
        self.subimage = None
        self.image_with_circle_msg = None

        #Circle variables
        self.cicle_msg = Circle()

        #Start align depth callback
        self.ready = False
        
        #Coordinates of circle
        self.circle_center_original_x = 0.0
        self.circle_center_original_y = 0.0

        # GetTF parameters
        self.TF_X = 0.0
        self.TF_Y = 0.0
        self.Z = 0.0

        # Parameter
        FILE = Path(__file__).resolve()
        ROOT = FILE.parents[0]
        if str(ROOT) not in sys.path:
            sys.path.append(str(ROOT))  # add ROOT to PATH
        ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


        self.declare_parameter('weights', self.weights_file)
        self.declare_parameter('data', self.data_file)
        self.declare_parameter('imagez_height', 640)
        self.declare_parameter('imagez_width', 640)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('max_det', 1000)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('view_img', True)
        self.declare_parameter('classes', None)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('half', False)
        self.declare_parameter('dnn', False)

        self.weights = self.get_parameter('weights').value
        self.data = self.get_parameter('data').value
        self.imagez_height = self.get_parameter('imagez_height').value
        self.imagez_width = self.get_parameter('imagez_width').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.device = self.get_parameter('device').value
        self.view_img = self.get_parameter('view_img').value
        self.classes = self.get_parameter('classes').value
        self.agnostic_nms = self.get_parameter('agnostic_nms').value
        self.line_thickness = self.get_parameter('line_thickness').value
        self.half = self.get_parameter('half').value
        self.dnn = self.get_parameter('dnn').value

        self.yolov5 = yolov5_demo(self.weights,
                                self.data,
                                self.imagez_height,
                                self.imagez_width,
                                self.conf_thres,
                                self.iou_thres,
                                self.max_det,
                                self.device,
                                self.view_img,
                                self.classes,
                                self.agnostic_nms,
                                self.line_thickness,
                                self.half,
                                self.dnn)


    #Callback 2 
    def aligned_depth_to_color_callback(self, image:Image):
        if self.ready:
            depth_image = self.bridge.imgmsg_to_cv2(image, "32FC1")
            depth_value = depth_image[int(self.cicle_msg.center_y), int(self.cicle_msg.center_x)]
            self.Z = depth_value/10
            self.get_logger().info(f"Depth value at circle center: {self.Z} cm")
            self.ready = False

            # Matriz intrínseca K
            K = [
                [390.72406005859375, 0.0, 323.9198913574219],
                [0.0, 390.72406005859375, 238.2157745361328],
                [0.0, 0.0, 1.0]
            ]

            # Coordenadas del punto en la imagen (u, v)
            u = self.circle_center_original_x 
            v = self.circle_center_original_y

            # Usando la inversa de la matriz intrínseca para obtener las coordenadas en el espacio tridimensional
            K_inv = np.linalg.inv(np.array(K))

            # Coordenadas homogéneas del punto en la imagen
            uv1 = np.array([[u], [v], [1.0]])

            # Multiplicación para obtener las coordenadas no escaladas
            xyz = np.dot(K_inv, uv1)

            # Escalar usando el valor Z para obtener las coordenadas finales (X, Y, Z)
            self.TF_X = xyz[0, 0] * self.Z
            self.TF_Y = xyz[1, 0] * self.Z

            self.get_logger().info(f"X: {self.TF_X}, Y: {self.TF_Y}, Z: {self.Z}")

    #def circle_sub_callback(self,msg):                    
    #    time.sleep(1.0)
    #    self.get_logger().info(f'Circle - Center: ({msg.center_x}, {msg.center_y}), Radius: {msg.radius}')

    def yolovFive2bboxes_msgs(self, bboxes:list, scores:list, cls:list, img_header:Header):
        bboxes_msg = BoundingBoxes()
        bboxes_msg.header = img_header

        print(bboxes)
        # print(bbox[0][0])
        i = 0
        for score in scores:
            one_box = BoundingBox()
            one_box.xmin = int(bboxes[0][i])
            one_box.ymin = int(bboxes[1][i])
            one_box.xmax = int(bboxes[2][i])
            one_box.ymax = int(bboxes[3][i])
            one_box.probability = float(score)
            one_box.class_id = cls[i]
            bboxes_msg.bounding_boxes.append(one_box)
            i = i+1

        return bboxes_msg

    #Callback 1
    def image_callback(self, image:Image):

        if self.start_img_callback:
            self.img_pub = image
            time.sleep(5.0)
            print("ready")
            image_raw = self.bridge.imgmsg_to_cv2(self.img_pub, "bgr8")
            class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list = self.yolov5.image_callback(image_raw)
            
            for xmin, ymin, xmax, ymax in zip(x_min_list, y_min_list, x_max_list, y_max_list):
                cv2.rectangle(image_raw, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            # Convierte la imagen modificada a un mensaje ROS
            self.image_with_bboxes = self.bridge.cv2_to_imgmsg(image_raw, "bgr8")

            self.msg = self.yolovFive2bboxes_msgs(bboxes=[x_min_list, y_min_list, x_max_list, y_max_list], scores=confidence_list, cls=class_list, img_header=self.img_pub.header)
            self.pub_bbox_.publish(self.msg)
            #self.pub_image.publish(self.img_pub)
            self.pub_image_.publish(self.image_with_bboxes)  # Publicar la imagen con los cuadros delimitadores

            print("start ==================")
            print(class_list, confidence_list, x_min_list, y_min_list, x_max_list, y_max_list)
            print("end ====================")

            # Coordenadas del primer cuadro delimitador
            xmin, ymin, xmax, ymax = map(int, [x_min_list[0], y_min_list[0], x_max_list[0], y_max_list[0]])
            
            # Extraer la subimagen dentro del primer cuadro delimitador
            self.subimage = image_raw[ymin:ymax, xmin:xmax]

            #Formato cv2
            form_image = self.subimage

            # Convierte la subimagen a un mensaje ROS si quieres publicarla
            self.subimage_msg = self.bridge.cv2_to_imgmsg(self.subimage, "bgr8")

            self.pub_subimage_.publish(self.subimage_msg)

            gray = cv2.cvtColor(form_image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    # Dibuja el círculo en la imagen
                    cv2.circle(form_image, (i[0], i[1]), i[2], (0,255,0), 2)
                    # Dibuja el centro del círculo
                    cv2.circle(form_image, (i[0], i[1]), 2, (0,0,255), 3)

                    # Calcula las coordenadas del centro del círculo en la imagen original
                    self.circle_center_original_x = i[0] + xmin
                    self.circle_center_original_y = i[1] + ymin

                    # Dibuja el círculo en la imagen original
                    cv2.circle(image_raw,(self.circle_center_original_x, self.circle_center_original_y), i[2], (0,0,255), 2)

                    # Publica los parámetros del círculo
                    self.cicle_msg.center_x = float(self.circle_center_original_x)
                    self.cicle_msg.center_y = float(self.circle_center_original_y)

                    self.cicle_msg.radius = float(i[2])
                    self.get_logger().info(
                     f'Circle - Center: ({self.cicle_msg.center_x}, {self.cicle_msg.center_y}), Radius: {self.cicle_msg.radius}')

                    self.circle_pub_.publish(self.cicle_msg)
                    #HERE


            # Convertir image_raw a un mensaje de ROS y publicarlo
            self.image_with_circle_msg = self.bridge.cv2_to_imgmsg(image_raw, "bgr8")
            self.pub_image_with_circle_pub_.publish(self.image_with_circle_msg )
            self.ready = True
            self.start_img_callback = False


###########################################################################################################################
###########################################################################################################################
################################### Obteniendo parametros #################################################################
###########################################################################################################################
###########################################################################################################################


def ros_main(args=None):
    rclpy.init(args=args)
    yolov5_node = yolov5_ros()

    executor = MultiThreadedExecutor()
    executor.add_node(yolov5_node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    yolov5_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    ros_main()