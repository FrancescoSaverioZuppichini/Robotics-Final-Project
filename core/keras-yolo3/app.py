from flask import Flask, request,jsonify
from yolo import YOLO
import numpy as np

yolo = YOLO()

app = Flask(__name__)

@app.route("/prediction", methods=['POST'])
def prediction():
    try:
        img = request.json['image']
        res, _ = yolo.predict(np.array(img))
    except Exception as e:
        res = {'error': str(e)}

    print(res)
    return jsonify(res)

@app.route("/model", methods=['GET'])
def get_classes():
    return jsonify(colors=yolo.colors, classes=yolo.class_names)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)