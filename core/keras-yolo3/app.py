from flask import Flask, request,jsonify, abort
from yolo import YOLO
import numpy as np
import base64
import cv2

yolo = YOLO()

app = Flask(__name__)

@app.route("/prediction", methods=['POST'])
def prediction():
    try:
        img = request.json['image']
        size = (416, 416)

        if 'size' in request.json.keys():
            size = (int(request.json['size'][0]), int(request.json['size'][1]))

        if 'compressed' in request.json.keys():
            img = np.fromstring(base64.b64decode(img), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        res, _ = yolo.predict(np.array(img), size=size)
        return jsonify(res)

    except Exception as e:
        abort(500, {'error': str(e)})


@app.route("/model", methods=['GET'])
def get_classes():
    return jsonify(colors=yolo.colors, classes=yolo.class_names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)