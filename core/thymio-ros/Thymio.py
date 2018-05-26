import rospy
import sys

import numpy as np
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from math import cos, sin, asin, tan, atan2
# msgs and srv for working with the set_model_service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from sensor_msgs.msg import Range, Image
from std_srvs.srv import Empty
from random import random

from utils import callback, Params

from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Thymio:

    def __init__(self, name, hooks=[]):
        """init"""
        self.name = name
        self.hooks = hooks
        self.sensors_cache = {}

        self.time_elapsed = 0
        self.start = 0

        rospy.init_node('basic_thymio_controller', anonymous=True)

        print(self.name)
        # Publish to the topic '/thymioX/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(self.name + '/cmd_vel',
                                                  Twist, queue_size=10)

        # A subscriber to the topic '/turtle1/pose'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber(self.name + '/odom',
                                                Odometry, self.update_state)
        self.sensors_names = ['rear_left','rear_right','left','center_left','center', 'center_right' ,'right']

        self.sensors_subscribers = [rospy.Subscriber(self.name + '/proximity/' + sensor_name,
        Range,
        callback(self.sensors_callback,i,sensor_name)) for i,sensor_name in enumerate(self.sensors_names)]

        self.camera_subscriber = rospy.Subscriber(self.name + '/camera/image_raw', Image, self.camera_callback, queue_size=1, buff_size=2**30)

        self.current_pose = Pose()
        self.current_twist = Twist()
        # publish at this rate
        self.rate = rospy.Rate(10)
        self.vel_msg = Twist()

    def sensors_callback(self, data, sensor_id, name):
        self.sensors_cache[name] = data

        try:
            for hook in self.hooks:
                hook.on_receive_sensor_data(self, data, sensor_id, name)
        except KeyError:
            pass

    def camera_callback(self, data):
        pass

    def thymio_state_service_request(self, position, orientation):
        """Request the service (set thymio state values) exposed by
        the simulated thymio. A teleportation tool, by default in gazebo world frame.
        Be aware, this does not mean a reset (e.g. odometry values)."""
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            model_state = ModelState()
            model_state.model_name = self.name
            model_state.reference_frame = '' # the frame for the pose information
            model_state.pose.position.x = position[0]
            model_state.pose.position.y = position[1]
            model_state.pose.position.z = position[2]
            qto = quaternion_from_euler(orientation[0], orientation[0], orientation[0], axes='sxyz')
            model_state.pose.orientation.x = qto[0]
            model_state.pose.orientation.y = qto[1]
            model_state.pose.orientation.z = qto[2]
            model_state.pose.orientation.w = qto[3]
            # a Twist can also be set but not recomended to do it in a service
            gms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            response = gms(model_state)

            return response
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def update_state(self, data):
        """A new Odometry message has arrived. See Odometry msg definition."""
        # Note: Odmetry message also provides covariance
        self.current_pose = data.pose.pose
        self.current_twist = data.twist.twist
        quat = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        (roll, pitch, yaw) = euler_from_quaternion (quat)

        # rospy.loginfo("State from Odom: (%.5f, %.5f, %.5f) " % (self.current_pose.position.x, self.current_pose.position.y, yaw))
        self.time_elapsed += 10
        for hook in self.hooks:
            hook.on_update_pose(self)

    def update_vel(self, linear, angular):
        vel_msg = Twist()

        vel_msg.linear.x = linear.x
        vel_msg.linear.y = linear.y
        vel_msg.linear.z = linear.z

        vel_msg.angular.x = angular.x
        vel_msg.angular.y = angular.y
        vel_msg.angular.z = angular.z

        self.vel_msg = vel_msg

    def move(self, linear, angular):
        """Moves the migthy thymio"""
        self.update_vel(linear, angular)

        while not rospy.is_shutdown():
            self.velocity_publisher.publish(self.vel_msg)
            self.rate.sleep()

        rospy.spin()

    def stop(self):
        self.update_vel(Params(), Params())