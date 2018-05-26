from Thymio import Thymio
import requests as req
import cv2
from pprint import pprint

from cv_bridge import CvBridge, CvBridgeError


class SmartThymio(Thymio, object):

    def __init__(self, *args, **kwargs):
        super(SmartThymio, self).__init__(*args, **kwargs)
        self.bridge = CvBridge()
        self.should_send = True # avoid automatic fire to the server
        self.draw = True

    def draw_image(self, cv_image):
        cv2.imshow('image', cv_image)
        cv2.waitKey(1)

    def camera_callback(self, data):
        if self.should_send:
            try:
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                self.should_send = False
                res = req.post('http://192.168.168.64:8080/prediction', json={'image' : image.tolist()})
                self.should_send = True
                pprint(res.json())

                if self.draw: self.draw_image(image)

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

