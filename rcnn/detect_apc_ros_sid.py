#!/usr/bin/env python

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import roslib
roslib.load_manifest('sub_py')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


CLASSES = ('__background__', # always index 0
        'barkely_bones', 'browns_brush', 'bunny_book', 'cherokee_tshirt',
        'clorox_brush', 'cloud_bear', 'command_hooks', 'cool_glue_sticks', 'crayola_24_ct',
        'creativity_stems', 'dasani_bottle', 'dove_bar', 'easter_sippy_cup',
        'elmers_school_glue', 'expo_eraser', 'fiskars_red',
        'fitness_dumbell', 'folgers_coffee', 'glucose_up_botle', 'hanes_socks',
        'jane_dvd', 'jumbo_pencil_cup', 'kleenex_towels', 'kygen_puppies', 'laugh_joke_book',
        'oral_green_toothbrush', 'oral_red_toothbrush', 'pencils', 'peva_liner',
        'platinum_bowl', 'rawlings_baseball', 'safety_plugs', 'scotch_mailer', 'scotch_tape',
        'staples_cards', 'tissue_box', 'while_lightbulb', 'womens_gloves', 'woods_cord')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        #print str(score) + '...' + str(class_name)

        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

        cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    

    '''im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        print score

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()'''

def demo(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

52    return args

##################################################################################

class image_converter:

  def __init__(self):

    self.image_pub = rospy.Publisher("logitech/webcam_raw_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("logitech/webcam_raw",Image,self.callback)
    print "komal"

  def callback(self,data):
    print "kalyan"
    
    cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    print(cv_image)
    
    demo(net, cv_image)
#    cv2.imwrite("/home/isl-server/Desktop/click2.jpg", cv_image)
#    cv2.imshow('output', cv_image)
    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    
    #cv2.imshow("Image window", cv_image)
    cv2.waitKey(1)

##################################################################################

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, '..', 'apc', 'faster_rcnn_test.pt')
    
    caffemodel = os.path.join(cfg.MODELS_DIR, '..', 'apc', 'vgg16_faster_rcnn_iter_70000.caffemodel')
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    rospy.spin()

