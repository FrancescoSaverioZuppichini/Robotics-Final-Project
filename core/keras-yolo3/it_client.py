import requests
import cv2
import numpy as np
from pprint import pprint
from yolo import YOLO

HOST_URL  = "http://localhost:8080"

res = requests.get('{}/model'.format(HOST_URL)).json()

colors, class_names = res['colors'], res['classes']

def unbox(res):
    res = res['res']

    out_scores = list(map(lambda x: x['score'], res))
    out_boxes = list(map(lambda x: x['boxes'], res))
    out_classes = list(map(lambda x: x['class'], res))
    out_classes_idx = list(map(lambda x: x['class_idx'], res))

    return out_scores, out_boxes, out_classes, out_classes_idx

while True:
    try:
        img_path = input('Input image path: ')
        src = cv2.imread(img_path)
        image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

        res = requests.post('{}/prediction'.format(HOST_URL), json = { 'image' : image.tolist(), 'size' : (416,416) })

        out_scores, out_boxes, out_classes, out_classes_idx = unbox(res.json())

        box_image = YOLO.draw_boxes(image, out_boxes, out_classes_idx, out_scores, colors, class_names, is_array=True)
        box_image.show()

        pprint(res.json())

    except Exception as e:
        print(e)
        continue