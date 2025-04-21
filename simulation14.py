#!/usr/bin/env python3        # use system’s python3 interpreter
import time                   # time utilities for delays and timestamps
import rclpy                  # ROS 2 Python client library
import cv2                    # OpenCV for image processing
import numpy as np            # numerical operations on arrays

from rclpy.node        import Node                                          # base class for ROS 2 nodes
from rclpy.qos         import QoSProfile, ReliabilityPolicy, HistoryPolicy  # QoS settings
from sensor_msgs.msg   import Image                                       # for subscribing to camera topic
from geometry_msgs.msg import Twist                                       # for sending velocity commands
from tello_msgs.srv    import TelloAction                                 # service for Tello takeoff/land
from cv_bridge         import CvBridge                                    # bridge ROS Image msgs ↔ OpenCV

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')                               # initialize node named "shape_flyer"
        self.bridge = CvBridge()                                       # converter between ROS images and OpenCV

        # FSM states
        self.state         = 'SEARCH'                                  # current behavior: SEARCH → ALIGN → FORWARD
        self.latest        = None                                      # last detected gate center + area
        self.frame         = None                                      # last camera frame for drawing & measurements

        # ALIGN gains + deadbands
        self.Kp_x, self.Kp_y = 0.5, 0.3                                # proportional gains for yaw (x) and altitude (y)
        self.dead_x, self.dead_y = 0.1, 0.1                            # tolerance window within which we consider “centered”

        # tuning
        self.search_yaw_speed = 0.3                                    # angular speed when spinning in SEARCH
        self.centered_frames  = 1                                      # number of consecutive centered frames before moving forward
        self.centered_count   = 0                                      # counter for consecutive centered frames

        # HSV thresholds for green gate detection
        self.green_low  = np.array((35,  80,  80))                     # lower bound of HSV for green
        self.green_high = np.array((85, 255, 255))                     # upper bound of HSV for green

        # subscribe & publish
        camera_qos = QoSProfile(                                       # configure QoS for camera subscription
            reliability=ReliabilityPolicy.BEST_EFFORT,                 # best effort: allow dropped frames
            history=   HistoryPolicy.KEEP_LAST,                        # keep only last N samples
            depth=     5                                               # queue depth
        )
        self.create_subscription(                                      # subscribe to the simulated drone’s camera
            Image,
            '/drone1/image_raw',
            self.image_cb,
            qos_profile=camera_qos
        )
        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)  # publisher for sending velocity commands

        # takeoff/land service client
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')  # create service client for Tello actions
        while not self.cli.wait_for_service(timeout_sec=1.0):                # wait until service is available
            self.get_logger().info('waiting for /drone1/tello_action...')
        self.takeoff()                                                      # send takeoff command on init

        # main loop @20Hz
        self.create_timer(1/20, self.control_loop)                           # call control_loop() at 20 Hz

    def takeoff(self):
        req = TelloAction.Request(); req.cmd='takeoff'   # build takeoff request
        self.cli.call_async(req); time.sleep(5.0)         # send it asynchronously, wait 5 s to climb

    def land(self):
        req = TelloAction.Request(); req.cmd='land'      # build land request
        self.cli.call_async(req)                         # send it

    def preprocess_mask(self, mask):
        """Clean noise then close small gaps, but leave ring edges for Hough."""
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))      # create 5×5 rectangular kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)   # remove small white noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)   # fill small holes in green regions
        return mask

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')                   # convert ROS Image to OpenCV BGR
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                   # convert BGR → HSV color space
        mask  = cv2.inRange(hsv, self.green_low, self.green_high)        # threshold to green
        mask  = self.preprocess_mask(mask)                               # clean up mask

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                   # grayscale version for circle detection
        blur = cv2.GaussianBlur(gray, (9,9), 2)                          # blur to reduce noise
        masked_gray = cv2.bitwise_and(blur, blur, mask=mask)             # apply green mask to blurred gray
        circles = cv2.HoughCircles(                                      # run Hough Circle Transform
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,                                                      # inverse ratio of accumulator resolution
            minDist=150,                                                 # minimum distance between circles
            param1=100,                                                  # higher threshold for Canny edge detector
            param2=25,                                                   # threshold for center detection
            minRadius=30,                                                # minimum radius of circles
            maxRadius=200                                                # maximum radius
        )

        if circles is not None:                                          # if any circles found
            x,y,r = circles[0][0]                                        # take the strongest detection
            cx, cy = int(x), int(y)                                     # convert center to int
            self.latest = {'center':(cx,cy), 'area':np.pi*r*r}          # estimate area of detection
            cv2.circle(frame, (cx,cy), int(r), (0,0,255), 2)            # draw circle outline
            cv2.circle(frame, (cx,cy),  4, (0,0,255), -1)               # draw center dot
        else:
            self.latest = self.detect_hole(frame, mask)                 # fallback to square‐hole detection

        self.frame = frame                                              # store last frame for control
        cv2.imshow('Gate Detector', frame)                              # show debug window
        cv2.waitKey(1)                                                  # needed for OpenCV window to update
        self.get_logger().info(self.latest and 'aligned' or 'no gate')  # log detection status

    def detect_hole(self, frame, mask):
        cnts, hier = cv2.findContours(mask,                             # retrieve contours with hierarchy
                                      cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return None                                                 # no contours → no gate
        hier = hier[0]                                                  # simplify hierarchy array

        holes = []
        for i,h in enumerate(hier):                                     # iterate through hierarchy entries
            if h[3] >= 0:                                               # h[3] ≥0 means contour has a parent → a hole
                c = cnts[i]                                             # corresponding contour
                area = cv2.contourArea(c)                              # compute area
                if area < 1500:                                        # ignore tiny holes
                    continue
                peri   = cv2.arcLength(c, True)                        # contour perimeter
                approx = cv2.approxPolyDP(c, 0.02*peri, True)          # polygonal approximation
                shape  = 'square' if len(approx)==4 else 'circle'     # classify by number of vertices
                holes.append((area,shape,c))                          # collect candidate holes

        if not holes:
            return None                                                 # no holes → no gate

        squares = [h for h in holes if h[1]=='square']                  # filter square holes
        if squares:
            best = max(squares, key=lambda x: x[0])                     # pick largest square if any
        else:
            best = max(holes,   key=lambda x: x[0])                     # else pick largest hole of any shape

        area, shape, c = best                                           # unpack best hole
        M = cv2.moments(c)                                              # compute image moments
        if M['m00']==0:
            return None                                                 # avoid division by zero
        cx = int(M['m10']/M['m00'])                                     # x‐centroid
        cy = int(M['m01']/M['m00'])                                     # y‐centroid
        color = (255,0,0) if shape=='square' else (0,0,255)             # choose draw color
        cv2.drawContours(frame,[c],-1,color,2)                          # draw hole contour
        cv2.circle(frame,(cx,cy),4,color,-1)                            # draw centroid
        return {'center':(cx,cy),'area':area}                          # return detection

    def control_loop(self):
        # SEARCH: hover until a gate appears
        if self.state=='SEARCH':
            if self.latest:                                             # gate detected?
                self.cmd_pub.publish(Twist())                           # stop any motion
                time.sleep(0.1)                                         # brief pause to settle
                self.state='ALIGN'                                      # switch to ALIGN state
                self.centered_count = 0                                 # reset centering counter
            else:
                self.cmd_pub.publish(Twist())                           # keep hovering
            return

        # ALIGN: center the detected gate
        if self.state=='ALIGN':
            if not self.latest:
                self.state='SEARCH'; return                            # lost sight → go back to SEARCH

            cx,cy = self.latest['center']                               # current gate center
            h,w   = self.frame.shape[:2]                                # frame dimensions
            err_x = (cx - w/2)/(w/2)                                    # normalized horizontal error
            err_y = (h/2 - cy)/(h/2)                                    # normalized vertical error

            cmd = Twist()                                               # prepare control command
            if abs(err_x)>self.dead_x:                                 # if x‐error outside deadband
                cmd.angular.z = -self.Kp_x*err_x                       # yaw to reduce horizontal error
            if abs(err_y)>self.dead_y:                                 # if y‐error outside deadband
                cmd.linear.z = self.Kp_y*err_y                        # climb/descend to reduce vertical error

            if abs(err_x)<=self.dead_x and abs(err_y)<=self.dead_y:    # if both errors small
                self.centered_count += 1                              # increment consecutive centered counter
            else:
                self.centered_count = 0                               # reset if error grows

            if self.centered_count >= self.centered_frames:            # if centered long enough
                self.cmd_pub.publish(Twist())                           # stop any residual motion
                time.sleep(0.1)                                         # pause to settle
                self.state='FORWARD'                                    # switch to forward state
            else:
                self.cmd_pub.publish(cmd)                              # otherwise keep aligning
            return

        # FORWARD: move straight as long as we still see the gate
        if self.state=='FORWARD':
            if self.latest:                                             # gate still in view?
                f = Twist(); f.linear.x=0.4                             # constant forward speed
                self.cmd_pub.publish(f)
            else:
                self.cmd_pub.publish(Twist())                           # gate gone → we’ve passed through
                time.sleep(0.1)                                         # pause
                self.state='SEARCH'                                     # look for next gate
            return

    def destroy_node(self):
        cv2.destroyAllWindows()                                       # close any OpenCV windows
        super().destroy_node()                                        # cleanup base Node

def main():
    rclpy.init()                                                     # initialize ROS 2 Python
    node = ShapeFlyer()                                              # create our node (takes off)
    try:
        rclpy.spin(node)                                             # enter spin loop (image_cb + control_loop)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()                                                  # on Ctrl‑C, tell drone to land
        rclpy.shutdown()                                             # shutdown ROS 2

if __name__=='__main__':
    main()                                                           # entry point
