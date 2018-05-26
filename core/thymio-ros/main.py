from Thymio import Thymio
from SmartThymio import SmartThymio
from utils import Params
import rospy

thymio = SmartThymio('thymio29')


thymio.stop()
rospy.sleep(1.)
thymio.move(Params(0),Params())
# rospy.sleep(1.)
