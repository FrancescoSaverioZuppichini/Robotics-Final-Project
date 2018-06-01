from Thymio import Thymio
from SmartThymio import SmartThymio
from utils import Params
from thymio_msgs.msg import Led
import rospy
import numpy as np

# thymio = Thymio('thymio29')

thymio = SmartThymio('thymio29')

rospy.sleep(1.)

thymio.interactive()
thymio.move(Params(0),Params())
