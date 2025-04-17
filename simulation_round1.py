#!/usr/bin/env python3
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

def detect_full_circle(img_bgr):
    """
    Thresholds a green ring, closes its interior, then fits
    a minimum enclosing circle. Returns center, radius, area,
    or None if no sufficiently large ring is found.
    """
    # 1) Blur → HSV
    blur = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 2) Green thresholds (tune these to your ring)
    lower = np.array([40, 100, 100])
    upper = np.array([80, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # 3) Close interior so we get one solid blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 4) Find contours, pick the largest
    cnts, _ = cv2.findContours(mask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c    = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 3000:  # filter out small noise
        return None

    # 5) Fit circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    cx, cy   = int(x), int(y)
    radius   = int(radius)

    return {'center': (cx, cy), 'radius': radius, 'area': area}

def compute_control(center, frame_shape, area):
    """
    Simple P‐controller: yaw + altitude + forward speed
    """
    h, w = frame_shape[:2]
    err_x = (center[0] - w/2) / (w/2)    # normalize to [-1,1]
    err_y = (h/2 - center[1]) / (h/2)    # positive → ring below center

    cmd = Twist()
    # yaw to reduce horizontal error
    cmd.angular.z = -0.5 * err_x
    # climb/descend to reduce vertical error
    cmd.linear.z  =  0.3 * err_y
    # slow forward as we near the ring
    target_area = 30000.0
    fwd = max(0.0, 1 - min(area/target_area, 1.0))
    cmd.linear.x  = 0.4 * fwd

    return cmd

class GateFlyerRing(Node):
    def __init__(self):
        super().__init__('gate_flyer_ring')
        self.bridge = CvBridge()
        self.state  = 'SEARCH'
        self.latest = None
        self.frame  = None

        # Subscriber (BEST_EFFORT for Gazebo camera)
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

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)

        # Takeoff / land service
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /drone1/tello_action...')
        self.takeoff()

        # Control loop @20Hz
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
        gate = detect_full_circle(img)
        self.frame  = img
        self.latest = gate

        # Overlay for debugging
        if gate:
            cx, cy = gate['center']
            r      = gate['radius']
            # circle outline
            cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
            # center dot
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        cv2.imshow('GateDetector', img)
        cv2.waitKey(1)

    def control_loop(self):
        if self.state == 'SEARCH':
            if self.latest:
                self.state = 'ALIGN'
            else:
                t = Twist()
                t.angular.z = 0.2    # spin rate
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
            # once centered & close → go THROUGH
            if (abs(cmd.angular.z) < 0.05 and
                abs(cmd.linear.z) < 0.05 and
                self.latest['area'] > 20000):
                self.state = 'THROUGH'
            return

        if self.state == 'THROUGH':
            t = Twist()
            t.linear.x = 0.5      # dash forward
            self.cmd_pub.publish(t)
            # when ring vanishes, back to SEARCH
            if not self.latest:
                self.state = 'SEARCH'
            return

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GateFlyerRing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
