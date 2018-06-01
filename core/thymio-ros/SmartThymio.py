from Thymio import Thymio
import requests as req
import cv2
from pprint import pprint
import requests
import rospy
from utils import draw_boxes, unbox, rect_area
import numpy as np
import json
from PID import PID
from cv_bridge import CvBridge, CvBridgeError
from utils import Params
from time import time
from time import sleep
import threading
from thymio_msgs.msg import Led
import sys
IMAGE_SAVE_DIR = './image_save/'

class SmartThymio(Thymio, object):

    def __init__(self, *args, **kwargs):
        self.sensors_cache_values = np.zeros(7)

        super(SmartThymio, self).__init__(*args, **kwargs)
        self.bridge = CvBridge()
        self.should_send = True # avoid automatic fire to the server
        self.draw = True
        self.HOST_URL = "http://192.168.168.64:8080"
        self.res = None
        self.global_step = 0
        self.camera_res  = (480,640)
        self.target = []
        self.angular_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.linear_pid = PID(Kd=5, Ki=0, Kp=0.5)
        self.object_pid = PID(Kd=3, Ki=0, Kp=0.5)

        self.last_elapsed = 0
        self.last_elapsed_sensors = 0

        self.MAX_TO_WAIT = 0.2
        self.FORWARD_VEL = 0.1

        self.p = threading.Thread()
        self.obstacle = False
        self.N_LEDS = 5
        self.colors, self.class_names = self.get_model_info()

    def get_model_info(self):
        res = requests.get('{}/model'.format(self.HOST_URL)).json()

        return res['colors'], res['classes']

    def draw_image(self, image, res, boxes=True, save=False):
        if boxes:
            out_scores, out_boxes, out_classes, out_classes_idx = unbox(res.json())

            pil_image = draw_boxes(image, out_boxes, out_classes_idx, out_scores, self.colors, self.class_names,
                                        is_array=True)

            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        if save:
            file_name = '{}'.format(self.global_step)
            cv2.imwrite(IMAGE_SAVE_DIR + file_name + '.jpg',image)
            np.save(IMAGE_SAVE_DIR  + file_name, image)
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

    def on_target_turn_on_leds(self, box):
        led_mask = np.zeros(self.N_LEDS)
        _, left, _, right = box
        height, width = self.camera_res

        start_led = int(((left / width) / 100) * self.N_LEDS * 100)
        end_led = int(((right / width) / 100) * self.N_LEDS * 100)

        led_mask[start_led:end_led + 1] = 1
        # remap the led to the correct Thymio's pins
        mask = np.concatenate([led_mask[2:], np.zeros(3), led_mask[:2]])

        self.turn_on_leds(mask)

    def turn_on_leds(self, mask=None, id=0):
        mask = np.zeros(8) if mask == None else mask

        self.led_subscriber.publish(Led(values=mask, id=id))

    def on_target(self, target_data):
        box = self.get_box(target_data)

        err, offset = self.get_error(box)

        height, width = self.camera_res
        err = err / width  # normalize in %
        dt = self.time_elapsed - self.last_elapsed
        ang_vel = self.object_pid.step(err, dt)

        # print('---------------')
        # print('err   : {:.2f}'.format(err))
        # print('offset: {:.2f}'.format(offset))
        # print('dt    : {:.2f}'.format(dt))
        # print('vel   : {:.2f}'.format(ang_vel))
        # print('Found {}').format(target_data['class'])

        self.last_elapsed = self.time_elapsed
        # print("{},".format(err))
        # ang_vel /= 10
        self.on_target_turn_on_leds(box)

        self.update_vel(Params(self.FORWARD_VEL), Params(z=-ang_vel))

    def interactive(self):
        def change_targets(thymio):
            while True:
                try:
                    classes = input("Change class:")
                    print(classes)
                    classes = classes.split(',')
                    thymio.target = classes
                    print "Targets changed to {}".format(thymio.target)
                except:
                    continue
        t = threading.Thread(target=change_targets, args=[self])
        t.start()

    def on_receive_sensor_data(self, data, sensor_id, name):
        val = data.range
        max = data.max_range

        if(val == np.inf): val = 0

        else:
            if(val < 0): val = data.min_range
            val = max - val
            val = val / max

        if sensor_id >= 5: val *= -1

        self.sensors_cache_values[sensor_id] = val

        self.obstacle = np.sum(self.sensors_cache_values) != 0

        if self.obstacle:

            lin_err = np.sum(self.sensors_cache_values) / self.sensors_cache_values.shape[0]
            ang_err = np.sum(self.sensors_cache_values[:2] - self.sensors_cache_values[3:5]) +  (self.sensors_cache_values[5] - self.sensors_cache_values[6])

            ang_vel = self.angular_pid.step(ang_err, 0.1)
            vel = self.linear_pid.step(lin_err, 0.1)

            self.last_elapsed_sensors = self.time_elapsed

            self.update_vel(Params(x=-vel), Params(z=-ang_vel))

    def explore(self):
        self.turn_on_leds()
        # rotate in place
        self.update_vel(Params(), Params(z=0.1))

    def select_target(self, targets, metric='closest'):
        if metric == 'closest':
            targets.sort(key=lambda x: -rect_area(x['boxes']))
        return targets[0]

    def on_targets(self, targets_data):
        target_data = self.select_target(targets_data)
        self.on_target(target_data)

    def on_prediction_success(self, pred):

        targets_data = self.find_targets_data_in_img(pred, self.target)

        there_are_targets = len(targets_data) > 0
        # select a target using some metrics, e.g rectangle area
        if not self.obstacle:
            if there_are_targets:
                self.on_targets(targets_data)
            else:
                # self.stop()
                self.explore()
        else:
            # print('Obstacle...')
            pass

    def ask_for_prediction(self, image):
        start = time()

        size  = (160,160)

        self.res = req.post('{}/prediction'.format(self.HOST_URL), json={'image': image.encode('base64'), 'size': size, 'compressed' : True})

        self.should_send = True

        self.global_step += 1

        pred = self.res.json()['res']

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
            if self.draw and self.res: self.draw_image(self.image, res=self.res, save=False)
        return

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




