#!/usr/bin/env python3

### this code detects the circle and then move forward 
import time
import rclpy
import cv2
import numpy as np

from rclpy.node        import Node
from rclpy.qos         import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg   import Image
from geometry_msgs.msg import Twist
from tello_msgs.srv    import TelloAction
from cv_bridge         import CvBridge

def detect_shape(img_bgr):
    """
    Detect the largest circle or square in the image regardless of color.
    Returns dict with center, area, shape type, or None if none found.
    """
    # 1) Preprocess: grayscale + blur + edge detection
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 2) Find contours
    cnts, _ = cv2.findContours(edges,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # 3) Examine contours by descending area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:
            break  # no larger shapes remain

        # 4) Approximate polygon for shape detection
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        shape = None
        if len(approx) == 4:
            shape = 'square'
        else:
            # test for circle by circularity
            (x, y), radius = cv2.minEnclosingCircle(c)
            circle_area = np.pi * radius * radius
            if circle_area > 0 and abs(circle_area - area) / circle_area < 0.25:
                shape = 'circle'

        if shape is None:
            continue

        # 5) Compute centroid
        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return {
            'center': (cx, cy),
            'area': area,
            'shape': shape,
            'contour': c
        }

    return None

def compute_control(center, frame_shape, area):
    """
    Simple P‐controller: yaw + vertical + forward
    """
    h, w = frame_shape[:2]
    err_x = (center[0] - w/2) / (w/2)
    err_y = (h/2 - center[1]) / (h/2)

    cmd = Twist()
    cmd.angular.z = -0.5 * err_x
    cmd.linear.z  =  0.3 * err_y

    # slow forward as shape grows
    tgt = 20000.0
    fwd = max(0.0, 1 - min(area / tgt, 1.0))
    cmd.linear.x = 0.4 * fwd

    return cmd

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')
        self.bridge = CvBridge()
        self.state  = 'SEARCH'
        self.latest = None
        self.frame  = None

        # QoS for Gazebo camera
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=   HistoryPolicy.KEEP_LAST,
            depth=     5
        )
        self.create_subscription(
            Image,
            '/drone1/image_raw',
            self.image_cb,
            qos_profile=camera_qos
        )

        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)

        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for tello_action...')
        self.takeoff()

        # run control loop at 20 Hz
        self.create_timer(1/20, self.control_loop)

    def takeoff(self):
        req = TelloAction.Request()
        req.cmd = 'takeoff'
        self.cli.call_async(req)
        time.sleep(5.0)

    def land(self):
        req = TelloAction.Request()
        req.cmd = 'land'
        self.cli.call_async(req)

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        shape = detect_shape(img)
        self.frame  = img
        self.latest = shape

        # overlay
        if shape:
            cx, cy = shape['center']
            if shape['shape'] == 'square':
                # draw bounding rect
                x,y,w,h = cv2.boundingRect(shape['contour'])
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            else:
                # circle overlay
                (x,y), r = cv2.minEnclosingCircle(shape['contour'])
                cv2.circle(img, (int(x),int(y)), int(r), (0,255,0), 2)
            cv2.circle(img, (cx,cy), 4, (0,0,255), -1)
            cv2.putText(img, shape['shape'], (cx-20, cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow('ShapeDetector', img)
        cv2.waitKey(1)

    def control_loop(self):
        if self.state == 'SEARCH':
            if self.latest:
                self.state = 'ALIGN'
            else:
                t = Twist()
                t.angular.z = 0.3
                self.cmd_pub.publish(t)
            return

        if self.state == 'ALIGN':
            if not self.latest:
                self.state = 'SEARCH'
                return
            cmd = compute_control(
                self.latest['center'],
                self.frame.shape,
                self.latest['area']
            )
            self.cmd_pub.publish(cmd)
            # when well aligned & close enough
            if (abs(cmd.angular.z) < 0.05 and
                abs(cmd.linear.z) < 0.05 and
                self.latest['area'] > 15000):
                self.state = 'THROUGH'
            return

        if self.state == 'THROUGH':
            t = Twist()
            t.linear.x = 0.5
            self.cmd_pub.publish(t)
            if not self.latest:
                self.state = 'SEARCH'
            return

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ShapeFlyer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
