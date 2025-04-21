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
        self.state         = 'SEARCH'                   #state: one of SEARCH → ALIGN → FORWARD.
        self.latest        = None                       #latest: stores last detected gate center & area.
        self.frame         = None                       #frame: stores last OpenCV frame for drawing & measurement.

        # ALIGN gains + deadbands
        self.Kp_x, self.Kp_y = 0.5, 0.3                 #Kp_x, Kp_y: proportional gains for yaw (horizontal) and z (vertical) control in ALIGN.
        self.dead_x, self.dead_y = 0.1, 0.1             #dead_x, dead_y: small thresholds below which we consider “centered” and stop jitter.

        # timing & counters
        self.search_yaw_speed = 0.3                     #search_yaw_speed: how fast to spin in SEARCH.
        self.forward_duration = 1.0                     #forward_duration: how many seconds to fly forward once aligned (guarantees passing).
        self.centered_frames  = 1                       #centered_frames: how many successive ALIGN frames must be “centered” before FORWARD. 
        self.centered_count   = 0                       #centered_count: counter for that.
        self.forward_start    = None                    #forward_start: timestamp when we began FORWARD.

        # green HSV range
        self.green_low  = np.array((35,  80,  80))
        self.green_high = np.array((85, 255, 255))

        # subscribe & publisher
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  #Best‐effort because we don’t need every frame, just as many as possible at high rate.
            history=   HistoryPolicy.KEEP_LAST,
            depth=     5
        )
        self.create_subscription(Image,
                                 '/drone1/image_raw',
                                 self.image_cb,          #We call self.image_cb on each new image.
                                 qos_profile=camera_qos)
        self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)  #Publisher to send velocity commands (Twist) to the drone.

        # takeoff/land service client
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')  #We create a ROS 2 service client for the Tello “action” interface.
        while not self.cli.wait_for_service(timeout_sec=1.0):               #We block until the service is available.
            self.get_logger().info('waiting for /drone1/tello_action...')   
        self.takeoff()                                                      #Then we call our helper self.takeoff() to launch the drone.

        # main loop
        self.create_timer(1/20, self.control_loop)


    def takeoff(self):
        req = TelloAction.Request(); req.cmd='takeoff'                       #Builds a TelloAction request with cmd='takeoff'.
        self.cli.call_async(req); time.sleep(5.0)                            #Sends it asynchronously and then sleep(5 s) to let it climb

    def land(self):
        req = TelloAction.Request(); req.cmd='land'                          #Similar for landing. Called at shutdown.
        self.cli.call_async(req)


    def preprocess_mask(self, mask):
        """Clean noise then close small gaps, but leave the ring edges for Hough."""
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))               #A small 5×5 rectangle of “on” pixels (all ones) that we’ll slide over the image.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)    #Removes small white speckles (noise) that aren’t big enough to survive a 5×5 erosion.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)    #Fills small black holes inside the detected green area—important when the ring’s paint or rendering has tiny breaks.
        return mask


    def image_cb(self, msg: Image):
        # 1) RGB → HSV → green mask
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')                     #Convert ROS image → OpenCV BGR.
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                     #Convert BGR → HSV.
        mask  = cv2.inRange(hsv, self.green_low, self.green_high)          #Threshold to green only.
        mask  = self.preprocess_mask(mask)                                 #Clean with morphological ops.

        # 2) Try Hough circle on the masked grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                     #Convert to gray → blur to reduce noise.
        blur = cv2.GaussianBlur(gray, (9,9), 2)                            #Apply the green mask to the blur.
        masked_gray = cv2.bitwise_and(blur, blur, mask=mask)
        circles = cv2.HoughCircles(                                         #Run Hough gradient to find circular shapes in that masked image.
            masked_gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,                                                        #dp=1.2: accumulator resolution.
            minDist=150,                                                   #minDist=150: require at least that many pixels between circle centers.
            param1=100,                                                    #param1/2: Canny edge threshold & Hough accumulator threshold.
            param2=25,
            minRadius=30,                                                  #minRadius/maxRadius: expected ring radii in pixels.
            maxRadius=200
        )

        if circles is not None:                                             #If at least one circle is found:
            # pick the strongest circle
            x,y,r = circles[0][0]                                           #We take the first (strongest) detection.
            cx, cy = int(x), int(y)                                         #Store its center (cx,cy) 
            self.latest = {'center':(cx,cy), 'area':np.pi*r*r}              #approximate pixel area πr² in self.latest.
            # draw it
            cv2.circle(frame, (cx,cy), int(r), (0,0,255), 2)                #Draw a red circle outline + center dot for debug.
            cv2.circle(frame, (cx,cy),  4, (0,0,255), -1)

        else:
            # fallback: square hole detection
            self.latest = self.detect_hole(frame, mask)                     #If no circle, we try to find square gates via contour‐hole logic.

        
        # show & store
        self.frame = frame
        cv2.imshow('Gate Detector', frame)                                  #Display the debug window.
        cv2.waitKey(1)
        self.get_logger().info(self.latest and 'aligned' or 'no gate')      #Log whether we have a detection this frame or not.


    def detect_hole(self, frame, mask):                                     
        """
        If no circle was found, try the old hole‐inside logic for squares.
        """
        cnts, hier = cv2.findContours(mask,                                 #RETR_CCOMP gives a contour hierarchy so it can find holes (inner contours).
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
                if area < 1500: continue                                    #We collect only those contours that have a parent in the hierarchy (i.e. holes inside a frame).
                peri   = cv2.arcLength(c,True)
                approx = cv2.approxPolyDP(c,0.02*peri,True)
                shape  = 'square' if len(approx)==4 else 'circle'           #Filter out tiny areas. Approximate their polygon to distinguish square vs circle.
                holes.append((area,shape,c))

        best = None
        # prefer circle holes, else square
        for area,shape,c in holes:
            if shape=='circle':
                best = (area,shape,c); break                                #Among all holes, we pick the first circle (if any), else the largest square.
            if not best:
                best = (area,shape,c)

        if not best:
            return None

        area,shape,c = best
        M = cv2.moments(c)                                                  #Compute the centroid of that hole via image moments.
        if M['m00']==0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # draw fallback overlay
        color = (0,0,255) if shape=='circle' else (255,0,0)
        cv2.drawContours(frame,[c],-1,color,2)                              #Draw it in the debug window (red for circle, blue for square).
        cv2.circle(frame,(cx,cy),4,color,-1)
        return {'center':(cx,cy),'area':area}                               #Return the same kind of {'center':..., 'area':...} dict.


    def control_loop(self):                                                 #Called at 20 Hz (we set that timer later). Drives the drone.
        # SEARCH
        if self.state=='SEARCH':                                            #If we don’t see any gate (self.latest is None):
            if self.latest:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)              #Publish a Twist() with only angular.z = +0.3 → spin in place searching. #As soon as we do see one:          #
                self.state = 'ALIGN'; self.centered_count=0                 #Publish an all‐zero Twist() to immediately stop that spin.Wait 0.1 s for it to settle.Switch to ALIGN state.
            else:                                                           
                t=Twist(); t.angular.z=self.search_yaw_speed
                self.cmd_pub.publish(t)
            return

        # ALIGN
        if self.state=='ALIGN':
            if not self.latest:
                self.state='SEARCH'; return                                 #If we lost sight of the gate, go back to SEARCH.

            cx,cy = self.latest['center']                                   #Compute normalized error in image space:
            h,w   = self.frame.shape[:2]
            err_x = (cx - w/2)/(w/2)                                        #err_x in [-1..1], negative ⇒ gate is to the left, positive ⇒ to the right.
            err_y = (h/2 - cy)/(h/2)                                        #err_y in [-1..1], negative ⇒ gate is too low, positive ⇒ too high.

            cmd=Twist()                                                     #If error is outside our deadband, we apply a P‐control: 
            if abs(err_x)>self.dead_x: cmd.angular.z = -self.Kp_x*err_x     #angular.z rotates the drone to keep the gate centered horizontally.
            if abs(err_y)>self.dead_y: cmd.linear.z  =  self.Kp_y*err_y     #linear.z climbs or descends to center vertically. 

            if abs(err_x)<=self.dead_x and abs(err_y)<=self.dead_y:         #We only switch to FORWARD once we’ve held both errors within deadband for N consecutive frames.
                self.centered_count += 1
            else:
                self.centered_count = 0

            if self.centered_count>=self.centered_frames:                   #If centered long enough:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)              #Send zero‐motion (Twist()) to stop any last-minute yaw/z drift.Sleep 0.1 s to settle.
                self.forward_start = time.time()                            #Record forward_start.
                self.state='FORWARD'                                        #Switch to FORWARD.
            else:
                self.cmd_pub.publish(cmd)
            return

        # FORWARD
        if self.state=='FORWARD':
            if time.time() - self.forward_start < self.forward_duration:    #For exactly forward_duration seconds (here 1 s):
                f=Twist(); f.linear.x=0.4; self.cmd_pub.publish(f)          #Publish a constant forward speed (linear.x = 0.4) to fly through the gate.
            else:                                                           #Once that timer expires:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)              #Send zero‐motion to stop.Sleep 0.1 s.
                self.state='SEARCH'                                         #Return to SEARCH to find the next gate.
            return


    def destroy_node(self):             
        cv2.destroyAllWindows()                                             #Closes any OpenCV windows on shutdown.
        super().destroy_node()


def main():
    rclpy.init()                                                            #Initializes the ROS 2 client library.
    node = ShapeFlyer()                                                     #Instantiates ShapeFlyer node (which immediately takes off).
    try:
        rclpy.spin(node)                                                    #spin(node) processes callbacks (image_cb and control_loop).
    except KeyboardInterrupt:
        pass
    finally:
        node.land()                                                         #On Ctrl‑C, land the drone, then cleanly shut down ROS 2.
        rclpy.shutdown()


if __name__=='__main__':
    main()
