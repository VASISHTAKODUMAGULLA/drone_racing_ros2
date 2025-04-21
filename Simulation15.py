#!/usr/bin/env python3
import time                                  # for sleep() and timing forward pulses
import rclpy                                 # ROS 2 Python client library
import cv2                                   # OpenCV for image processing
import numpy as np                           # numerical library for arrays

# ROS 2 message and service types:
from rclpy.node        import Node           # base class for ROS 2 nodes
from rclpy.qos         import QoSProfile, ReliabilityPolicy, HistoryPolicy  
                                              # for fine‑tuning subscriber QoS
from sensor_msgs.msg   import Image          # camera image message
from geometry_msgs.msg import Twist          # velocity command message
from tello_msgs.srv    import TelloAction    # Tello takeoff/land service
from cv_bridge         import CvBridge       # convert between ROS 2 images and OpenCV
import cv2.aruco as aruco

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')        # initialize node with name "shape_flyer"
        self.bridge = CvBridge()               # helper to convert ROS images ↔ OpenCV

        # ─── Finite‑State‑Machine state ───
        self.state = 'SEARCH'                  # current state: SEARCH → ALIGN → FORWARD
        self.latest = None                     # most recent detection dict or None
        self.frame  = None                     # last BGR OpenCV image for drawing

        # ─── ALIGN controller gains & deadbands ───
        self.Kp_x, self.Kp_y = 0.5, 0.3         # P‑gain for yaw (x error) and z (y error)
        self.dead_x, self.dead_y = 0.05, 0.05     # error threshold to consider “centered”

        # ─── Timing parameters & counters ───
        self.search_yaw_speed = 0.1            # angular.z speed while spinning in SEARCH
        self.centered_frames  = 3              # number of consecutive frames within deadband
        self.centered_count   = 0              # counter for how many frames we’ve been centered
        self.forward_start    = None           # timestamp when we entered FORWARD

        # ─── Green HSV threshold for gate masking ───
        self.green_low  = np.array((35,  80,  80))   # lower HSV bound for “green”
        self.green_high = np.array((85, 255, 255))   # upper HSV bound for “green”

        # ─── ArUco dictionary & detection parameters ───
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()

        # ─── Image subscriber (best‑effort, keep‑last) ───
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # drop old frames under load
            history=   HistoryPolicy.KEEP_LAST,         # only store last N
            depth=     5                                # queue depth
        )
        self.create_subscription(
            Image,
            '/drone1/image_raw',
            self.image_cb,                              # callback on each incoming image
            qos_profile=camera_qos
        )

        # ─── Velocity command publisher ───
        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)

        # ─── Tello takeoff/land service client ───
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            # wait until TelloAction service becomes available
            self.get_logger().info('waiting for /drone1/tello_action…')
        self.takeoff()                                # automatically take off on startup

        # ─── Main control loop at 20 Hz ───
        self.create_timer(1/20, self.control_loop)


    def takeoff(self):
        """Command Tello to take off and sleep to allow it to climb."""
        req = TelloAction.Request()                   # create service request
        req.cmd = 'takeoff'                           # set command field
        self.cli.call_async(req)                      # send asynchronously
        time.sleep(5.0)                               # wait 5 seconds for safe climb


    def land(self):
        """Command Tello to land."""
        req = TelloAction.Request()
        req.cmd = 'land'
        self.cli.call_async(req)


    def image_cb(self, msg: Image):
        """
        Called each time a new Image arrives.
        1) Try green gate detection
        2) If none, try fiducial
        3) If still none, try STOP sign
        """
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')  # convert ROS image → OpenCV BGR
        self.frame = img                              # store for control_loop

        # 1) GREEN GATE detection (ring, filled square, hollow frame)
        gate = self.detect_green_gate(img)
        if gate is None:
            # 2) if no green gate, try ArUco fiducial
            gate = self.detect_fiducial(img)

        if gate is None:
            # 3) if still none, check STOP sign
            stop = self.detect_stop(img)
            if stop:
                self.get_logger().warn('STOP detected → landing')
                self.cmd_pub.publish(Twist())        # cease any motion
                self.land()                          # land immediately
                rclpy.shutdown()                    # shutdown ROS 2
                return

        # store whichever (gate or None) we found this frame
        self.latest = gate

        # show debug window & log
        cv2.imshow('ShapeFlyer View', img)
        cv2.waitKey(1)
        self.get_logger().info(
            gate and f"see {gate['shape']} area={int(gate['area'])}" or "no gate"
        )


    def detect_stop(self, img):
        """
        Detect a red octagon (STOP sign):
        - threshold red in two HSV slices
        - find large contours with 8 vertices
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # red Hue wraps around, so use two masks:
        l1,h1 = np.array((0,70,50)),   np.array((10,255,255))
        l2,h2 = np.array((170,70,50)), np.array((180,255,255))
        m1 = cv2.inRange(hsv, l1, h1)
        m2 = cv2.inRange(hsv, l2, h2)
        red = cv2.bitwise_or(m1, m2)                # combine both red slices

        # close small holes in mask
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, k, iterations=2)

        cnts, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            A = cv2.contourArea(c)
            if A < 2000:                            # ignore tiny patches
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 8:                    # eight‐sided?
                M = cv2.moments(c)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # draw for debug
                cv2.drawContours(img, [approx], -1, (0,0,255), 2)
                cv2.circle(img, (cx,cy), 4, (0,0,255), -1)
                return {'center':(cx,cy),'area':A,'shape':'stop'}
        return None


    def detect_green_gate(self, img):
        """
        Detect green gates in three passes:
         1) rings via HoughCircles
         2) filled squares via 4‑vertex contours
         3) hollow frames via parent/child contour logic
        """
        # A) build clean green mask
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_low, self.green_high)
        k    = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        # B) ring detection: HoughCircles on masked blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)
        mg   = cv2.bitwise_and(blur, blur, mask=mask)
        circles = cv2.HoughCircles(
            mg, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=100,
            param1=100, param2=25,
            minRadius=20, maxRadius=200
        )
        if circles is not None:
            x,y,r = circles[0][0]
            cx,cy = int(x), int(y)
            A = np.pi*r*r
            # draw circle
            cv2.circle(img, (cx,cy), int(r), (0,255,0), 2)
            cv2.circle(img, (cx,cy), 4, (0,255,0), -1)
            return {'center':(cx,cy),'area':A,'shape':'circle'}

        # C) contour‐based for squares & frames
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return None
        hier = hier[0]

        filled, frames = [], []
        for i,hinfo in enumerate(hier):
            c = cnts[i]
            A = cv2.contourArea(c)
            if A < 2000:
                continue
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            # if it has a child hole → hollow frame
            if any(h[3] == i for h in hier):
                frames.append((A,approx,c))
            else:
                filled.append((A,approx,c))

        # 1) biggest filled square?
        best_sq = max((f for f in filled if len(f[1])==4),
                      default=None, key=lambda x:x[0])
        if best_sq:
            A, approx, c = best_sq
            M = cv2.moments(c)
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            x,y,wc,hc = cv2.boundingRect(c)
            cv2.rectangle(img, (x,y),(x+wc,y+hc), (255,255,0), 2)
            cv2.circle(img, (cx,cy),4,(255,255,0), -1)
            return {'center':(cx,cy),'area':A,'shape':'square'}

        # 2) largest hollow frame (square or circle)
        if frames:
            A, approx, c = max(frames, key=lambda x:x[0])
            (x0,y0),r0 = cv2.minEnclosingCircle(c)
            circA = np.pi*r0*r0
            # choose shape by circularity
            shape = 'circle' if abs(circA-A)/circA<0.3 else 'square'
            M = cv2.moments(c)
            if M['m00'] == 0:
                return None
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            col = (0,255,255) if shape=='circle' else (255,0,255)
            cv2.drawContours(img, [c], -1, col, 2)
            cv2.circle(img,(cx,cy),4,col,-1)
            return {'center':(cx,cy),'area':A,'shape':f'frame_{shape}'}

        return None


    def detect_fiducial(self, img):
        """
        Fall back to ArUco fiducial if no green gate found.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return None
        c = corners[0][0]                         # take first marker’s 4 corners
        cx = int(c[:,0].mean()); cy = int(c[:,1].mean())
        A  = cv2.contourArea(c.astype(np.int32))
        cv2.polylines(img, [c.astype(int)], True, (0,128,255), 2)
        cv2.circle(img,(cx,cy),4,(0,128,255),-1)
        return {'center':(cx,cy),'area':A,'shape':'fiducial'}


    def control_loop(self):
        """
        Called at 20 Hz:
         - SEARCH: spin until we see gate → ALIGN
         - ALIGN: center horizontally & vertically → FORWARD
         - FORWARD: fly straight until gate vanishes → SEARCH
        """
        # ─── SEARCH ───
        if self.state == 'SEARCH':
            if self.latest:
                # found something → stop spin, reset counter, go ALIGN
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)
                self.state = 'ALIGN'
                self.centered_count = 0
            else:
                # keep spinning in place
                t = Twist(); t.angular.z = self.search_yaw_speed
                self.cmd_pub.publish(t)
            return

        # ─── ALIGN ───
        if self.state == 'ALIGN':
            if not self.latest:
                # lost sight → back to SEARCH
                self.state = 'SEARCH'
                return
            cx,cy = self.latest['center']            # image coordinates
            h,w   = self.frame.shape[:2]
            err_x = (cx - w/2)/(w/2)                  # normalized x error
            err_y = (h/2 - cy)/(h/2)                  # normalized y error
            cmd   = Twist()
            # apply proportional yaw
            if abs(err_x) > self.dead_x:
                cmd.angular.z = -self.Kp_x * err_x
            # apply proportional climb/descend
            if abs(err_y) > self.dead_y:
                cmd.linear.z  =  self.Kp_y * err_y
            # count stable frames
            if abs(err_x) <= self.dead_x and abs(err_y) <= self.dead_y:
                self.centered_count += 1
            else:
                self.centered_count = 0
            # once centered long enough, switch to FORWARD
            if self.centered_count >= self.centered_frames:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                self.forward_start = time.time()
                self.state = 'FORWARD'
            else:
                self.cmd_pub.publish(cmd)
            return

        # ─── FORWARD ───
        if self.state == 'FORWARD':
            # continue forward while gate still visible and under 3s
            if self.latest :
                f = Twist(); f.linear.x = 0.1
                self.cmd_pub.publish(f)
            else:
                # done passing → stop & return to SEARCH
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                self.state = 'SEARCH'
            return


    def destroy_node(self):
        """Close OpenCV windows on node destruction."""
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()               # initialize ROS 2 Python
    node = ShapeFlyer()        # create node (it takes off automatically)
    try:
        rclpy.spin(node)       # spin to process callbacks
    except KeyboardInterrupt:
        pass
    finally:
        node.land()            # ensure we land on exit
        rclpy.shutdown()       # clean shutdown

if __name__ == '__main__':
    main()
