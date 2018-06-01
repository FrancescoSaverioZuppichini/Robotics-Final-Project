# Object Detection-Based Behaviour using Deep-Learning on Thymio
### Francesco Saverio Zuppichini and Alessia Ruggeri

## Installation
The guide is divided in two parts, one for YOLO and one for ROS

### Yolo
Download the model weights from:

https://drive.google.com/open?id=1UGl-POxop4SaEUTTFovzL3x81-FHvFbq

and paste the whole directory into `./core/keras-yolo3`.

Start the web server by running `python3 app.py`. You server will be accesible at `localhost:8008`

### ROS
You need to install ROS. Connect to the Tymio's wifi and run `python ./core/thymio-ros/main.py`.

