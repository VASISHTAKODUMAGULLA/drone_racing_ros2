#!/usr/bin/env python3

##better version of simulation2
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
    Detect the largest circle or square in the image.
    Returns {'center':(cx,cy),'area':area,'shape':..., 'contour':...} or None.
    """
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < 2000:
            break

        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        shape  = None
        if len(approx) == 4:
            shape = 'square'
        else:
            (x, y), r = cv2.minEnclosingCircle(c)
            circ_area = np.pi * r * r
            if circ_area > 0 and abs(circ_area - area) / circ_area < 0.25:
                shape = 'circle'

        if not shape:
            continue

        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return {'center': (cx, cy), 'area': area, 'shape': shape, 'contour': c}

    return None

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')
        self.bridge = CvBridge()
        self.state  = 'SEARCH'
        self.latest = None
        self.frame  = None

        # PD gains for ALIGN
        self.Kp_x, self.Kp_y = 0.5, 0.3
        self.dead_x, self.dead_y = 0.05, 0.05

        # QoS for Gazebo camera
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=   HistoryPolicy.KEEP_LAST,
            depth=     5
        )
        self.create_subscription(
            Image, '/drone1/image_raw', self.image_cb, qos_profile=camera_qos
        )
        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)

        # takeoff/land service
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for tello_action...')
        self.takeoff()

        # main loop @20Hz
        self.create_timer(1/20, self.control_loop)

    def takeoff(self):
        req = TelloAction.Request(); req.cmd = 'takeoff'
        self.cli.call_async(req); time.sleep(5.0)

    def land(self):
        req = TelloAction.Request(); req.cmd = 'land'
        self.cli.call_async(req)

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.latest = detect_shape(img)
        self.frame  = img

        # draw overlay
        if self.latest:
            cx, cy = self.latest['center']
            if self.latest['shape']=='square':
                x,y,w,h = cv2.boundingRect(self.latest['contour'])
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            else:
                (x,y),r = cv2.minEnclosingCircle(self.latest['contour'])
                cv2.circle(img,(int(x),int(y)),int(r),(0,255,0),2)
            cv2.circle(img,(cx,cy),4,(0,0,255),-1)
            cv2.putText(img,self.latest['shape'],(cx-20,cy-20),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow('Shape', img)
        cv2.waitKey(1)

    def control_loop(self):
        # --- SEARCH ---
        if self.state=='SEARCH':
            if self.latest:
                # stop any rotation immediately
                stop = Twist(); self.cmd_pub.publish(stop)
                time.sleep(0.1)
                self.state='ALIGN'
            else:
                spin = Twist(); spin.angular.z=0.3
                self.cmd_pub.publish(spin)
            return

        # --- ALIGN ---
        if self.state=='ALIGN':
            if not self.latest:
                self.state='SEARCH'; return

            cx,cy = self.latest['center']
            h,w = self.frame.shape[:2]
            err_x = (cx - w/2)/(w/2)
            err_y = (h/2 - cy)/(h/2)

            cmd = Twist()
            # yaw if outside deadband
            if abs(err_x)>self.dead_x:
                cmd.angular.z = -self.Kp_x*err_x
            # vertical if outside deadband
            if abs(err_y)>self.dead_y:
                cmd.linear.z  =  self.Kp_y*err_y

            # if both errors within deadband, transition to FORWARD
            if abs(err_x)<=self.dead_x and abs(err_y)<=self.dead_y:
                # stop all rotation/vertical
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                time.sleep(0.1)
                self.state='FORWARD'
            else:
                self.cmd_pub.publish(cmd)
            return

        # --- FORWARD ---
        if self.state=='FORWARD':
            if not self.latest:
                self.state='SEARCH'; return

            fwd = Twist()
            fwd.linear.x = 0.4   # constant forward
            self.cmd_pub.publish(fwd)
            return

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = ShapeFlyer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()
        rclpy.shutdown()

if __name__=='__main__':
    main()
