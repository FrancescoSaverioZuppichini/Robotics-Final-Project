from Thymio import Thymio
from SmartThymio import SmartThymio
from utils import Params
from thymio_msgs.msg import Led
import rospy
import numpy as np

# thymio = Thymio('thymio29')

thymio = SmartThymio('thymio29')

# thymio.stop()
rospy.sleep(1.)
# mask = np.array([1.0,1.0,1.0,0,0,0,1,1])
# thymio.led_subscriber.publish(Led(values=np.zeros(8), id=0))
# thymio.led_subscriber.publish(Led(values=mask, id=0))
thymio.move(Params(0),Params())
