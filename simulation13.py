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

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')
        self.bridge = CvBridge()

        # FSM states
        self.state         = 'SEARCH'
        self.latest        = None
        self.frame         = None

        # ALIGN gains + deadbands
        self.Kp_x, self.Kp_y = 0.5, 0.3
        self.dead_x, self.dead_y = 0.1, 0.1

        # timing & counters
        self.search_yaw_speed = 0.3
        self.forward_duration = 1.0
        self.centered_frames  = 1
        self.centered_count   = 0
        self.forward_start    = None

        # green HSV range
        self.green_low  = np.array((35,  80,  80))
        self.green_high = np.array((85, 255, 255))

        # subscribe & publisher
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

        # takeoff/land service client
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /drone1/tello_action...')
        self.takeoff()

        # main loop
        self.create_timer(1/20, self.control_loop)


    def takeoff(self):
        req = TelloAction.Request(); req.cmd='takeoff'
        self.cli.call_async(req); time.sleep(5.0)

    def land(self):
        req = TelloAction.Request(); req.cmd='land'
        self.cli.call_async(req)


    def preprocess_mask(self, mask):
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)
        return mask


    def image_cb(self, msg: Image):
        # 1) RGB → HSV → green mask
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, self.green_low, self.green_high)
        mask  = self.preprocess_mask(mask)

        # 2) Try Hough circle on the masked grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9,9), 2)
        masked_gray = cv2.bitwise_and(blur, blur, mask=mask)
        circles = cv2.HoughCircles(
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=150,
            param1=100,
            param2=25,
            minRadius=30,
            maxRadius=200
        )

        if circles is not None:
            x,y,r = circles[0][0]
            cx, cy = int(x), int(y)
            self.latest = {'center':(cx,cy), 'area':np.pi*r*r}
            cv2.circle(frame, (cx,cy), int(r), (0,0,255), 2)
            cv2.circle(frame, (cx,cy),  4, (0,0,255), -1)
        else:
            self.latest = self.detect_hole(frame, mask)

        self.frame = frame
        cv2.imshow('Gate Detector', frame)
        cv2.waitKey(1)
        self.get_logger().info(self.latest and 'aligned' or 'no gate')


    def detect_hole(self, frame, mask):
        cnts, hier = cv2.findContours(mask,
                                      cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return None
        hier = hier[0]

        holes = []
        for i,h in enumerate(hier):
            if h[3] >= 0:
                c = cnts[i]
                area = cv2.contourArea(c)
                if 1000 < area < 1500:
                    continue
                peri   = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02*peri, True)
                shape  = 'square' if len(approx)==4 else 'circle'
                holes.append((area,shape,c))

        if not holes:
            return None

        # prefer square over circle
        squares = [h for h in holes if h[1]=='square']
        if squares:
            best = max(squares, key=lambda x: x[0])
        else:
            best = max(holes, key=lambda x: x[0])

        area, shape, c = best
        M = cv2.moments(c)
        if M['m00']==0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        color = (255,0,0) if shape=='square' else (0,0,255)
        cv2.drawContours(frame,[c],-1,color,2)
        cv2.circle(frame,(cx,cy),4,color,-1)
        return {'center':(cx,cy),'area':area}


    def control_loop(self):
        # SEARCH: hover until we see a gate
        if self.state=='SEARCH':
            if self.latest:
                # we found it, stop and go align
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)
                self.state='ALIGN'
                self.centered_count=0
            else:
                # NO OBSTACLE → hover in place (do NOT move forward or spin)
                self.cmd_pub.publish(Twist())
            return

        # ALIGN: center the gate in frame
        if self.state=='ALIGN':
            if not self.latest:
                self.state='SEARCH'
                return
            cx,cy = self.latest['center']
            h,w   = self.frame.shape[:2]
            err_x = (cx - w/2)/(w/2)
            err_y = (h/2 - cy)/(h/2)

            cmd = Twist()
            if abs(err_x)>self.dead_x:
                cmd.angular.z = -self.Kp_x*err_x
            if abs(err_y)>self.dead_y:
                cmd.linear.z = self.Kp_y*err_y

            if abs(err_x)<=self.dead_x and abs(err_y)<=self.dead_y:
                self.centered_count += 1
            else:
                self.centered_count = 0

            if self.centered_count>=self.centered_frames:
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)
                self.forward_start = time.time()
                self.state='FORWARD'
            else:
                self.cmd_pub.publish(cmd)
            return

        # FORWARD: fly straight through then hover
        if self.state=='FORWARD':
            if time.time() - self.forward_start < self.forward_duration:
                f=Twist(); f.linear.x=0.4; self.cmd_pub.publish(f)
            else:
                # done passing → hover and look for next
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)
                self.state='SEARCH'
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
