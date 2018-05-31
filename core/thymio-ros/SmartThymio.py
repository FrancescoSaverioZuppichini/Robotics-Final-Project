from Thymio import Thymio
import requests as req
import cv2
from pprint import pprint
import requests
import rospy
from utils import draw_boxes, unbox
import numpy as np
import json
from PID import PID
from cv_bridge import CvBridge, CvBridgeError
from utils import Params
from time import time
from time import sleep
import threading

IMAGE_SAVE_DIR = './image_save/'

class SmartThymio(Thymio, object):

    def __init__(self, *args, **kwargs):
        super(SmartThymio, self).__init__(*args, **kwargs)
        self.bridge = CvBridge()
        self.should_send = True # avoid automatic fire to the server
        self.draw = True
        self.HOST_URL = "http://192.168.168.64:8080"
        res = requests.get('{}/model'.format(self.HOST_URL)).json()
        self.colors, self.class_names = res['colors'], res['classes']
        self.global_step = 0
        self.camera_res  = (480,640)
        self.target = ['dog','teddy bear']
        self.angular_pid = PID(Kd=5, Ki=0, Kp=1)
        self.last_elapsed = 0
        self.MAX_TO_WAIT = 0.2
        self.p = threading.Thread()

    def draw_image(self, image, res, boxes=True, save=False):

        if boxes:
            out_scores, out_boxes, out_classes, out_classes_idx = unbox(res.json())

            pil_image = draw_boxes(image, out_boxes, out_classes_idx, out_scores, self.colors, self.class_names,
                                        is_array=True)

            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if save:
            file_name = '{}'.format(self.global_step)
            np.save(IMAGE_SAVE_DIR + file_name, image)
            with open('{}{}.json'.format(IMAGE_SAVE_DIR,file_name), 'w') as f:
                json.dump(res.json(), f)

        cv2.imshow('image', image)
        cv2.waitKey(1)

    def find_targets_data_in_img(self, data, target):
        targets = list(filter(lambda x: x['class'] in target, data))
        return targets

    def get_class_data(seldf, data, class_name):
        filtered = list(filter(lambda x: x['class'] == class_name, data))

        return None if len(filtered) <= 0 else filtered[0]

    def get_box(self, data):
        box = np.array(data['boxes'])
        box[box < 0] = 0  # prune negative
        return box


    def get_error(self, box):
        height, width = self.camera_res
        img_mid_p = width // 2
        top, left, bottom, right = box
        target_mid_p = np.abs(left - right) / 2
        offset = (left + target_mid_p)
        err = offset - img_mid_p
        return err, offset

    def on_prediction_success(self, pred):

        targets_data = self.find_targets_data_in_img(pred, self.target)

        there_are_targets = len(targets_data) > 0
        # select a target using some metrics, e.g rectangle area
        if there_are_targets:
            target_data = targets_data[0]

            box = self.get_box(target_data)

            err, offset = self.get_error(box)

            width, _ = self.camera_res
            err = err / width # normalize in %
            dt = self.time_elapsed - self.last_elapsed
            ang_vel = self.angular_pid.step(err, dt)

            # pprint(box)

            print('err   : {:.2f}'.format(err))
            print('offset: {:.2f}'.format(offset))
            print('dt    : {:.2f}'.format(dt))
            print('vel   : {:.2f}'.format(ang_vel))

            self.last_elapsed =  self.time_elapsed
            # ang_vel /= 10

            self.update_vel(Params(), Params(z=-1 * ang_vel))
            print(ang_vel)
        else:
            # should go in exploration mode
            self.stop()

    def ask_for_prediction(self, image):
        start = time()

        size  = (160,160)

        self.res = req.post('{}/prediction'.format(self.HOST_URL), json={'image': image.encode('base64'), 'size': size, 'compressed' : True})

        self.should_send = True

        self.global_step += 1

        pred = self.res.json()['res']

        # pprint(pred)

        self.on_prediction_success(pred)

        end = time()

        # print('Prediction took {:.4f}'.format(end - start))

        return self.res

    def on_get_image_from_camera_success(self, image):

        self.p = threading.Thread(target=self.ask_for_prediction, args=[image])
        self.p.start()

    def camera_callback_compressed(self, data):
        should_send = not self.p.is_alive()

        if should_send:
            try:
                compressed = data.data
                np_arr = np.fromstring(data.data, dtype=np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.on_get_image_from_camera_success(compressed)
                self.image = image

            except Exception as e:
            #     # TODO add connection error and handling
                print(e)
        else:
            if self.draw and self.res: self.draw_image(self.image, res=self.res)


    def camera_callback(self, data):
        if self.should_send:
            try:
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.on_get_image_from_camera_success(image)
                self.image = image

            except CvBridgeError as e:
                print(e)
                print('Could not convert to cv2')
            except Exception as e:
            #     # TODO add connection error and handling
                print(e)
        else:
            if self.draw and self.res: self.draw_image(self.image, res=self.res)




