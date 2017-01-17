#!/usr/bin/env python
import roslib
roslib.load_manifest('sub_py')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):

    self.image_pub = rospy.Publisher("logitech/webcam_raw_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("logitech/webcam_raw",Image,self.callback)

##########################################################
  def callback(self,data):
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)

    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

###########################################################
def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  rospy.spin()
  cv2.destroyAllWindows()

###########################################################
if __name__ == '__main__':
    main(sys.argv)
