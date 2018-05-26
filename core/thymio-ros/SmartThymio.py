from Thymio import Thymio
import requests as req
import cv2
from pprint import pprint
import requests
import rospy
from utils import draw_boxes, unbox
import numpy as np
import json

from cv_bridge import CvBridge, CvBridgeError

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
        print("got them!")

    def draw_image(self, image, res, boxes=True, save=True):

        if boxes:
            out_scores, out_boxes, out_classes, out_classes_idx = unbox(res.json())

            pil_image = draw_boxes(image, out_boxes, out_classes_idx, out_scores, self.colors, self.class_names,
                                        is_array=True)

            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if save:
            file_name = 'image_{}'.format(self.global_step)
            np.save(IMAGE_SAVE_DIR + file_name, image)
            with open('{}{}.json'.format(IMAGE_SAVE_DIR,file_name), 'w') as f:
                json.dump(res.json(), f)

        cv2.imshow('image', image)
        cv2.waitKey(1)

    def camera_callback(self, data):
        if self.should_send:
            try:
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.should_send = False
                res = req.post('{}/prediction'.format(self.HOST_URL), json={'image' : image.tolist()})
                self.should_send = True
                pprint(res.json())

                if self.draw: self.draw_image(image, res=res)
                self.global_step += 1

            except CvBridgeError as e:
                print(e)
                print('Could not convert to cv2')
            except Exception as e:
                # TODO add connection error and handling
                print(e)
                print('Something exploded!!')
        else:
            print 'Skipped!'
        return

