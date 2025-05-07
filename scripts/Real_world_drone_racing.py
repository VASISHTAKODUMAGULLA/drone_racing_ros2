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
import cv2.aruco as aruco
import pytesseract

class ShapeFlyer(Node):
    def __init__(self):
        super().__init__('shape_flyer')
        self.bridge = CvBridge()

        # FSM states
        self.state         = 'SEARCH'                   #state: one of SEARCH → ALIGN → FORWARD.
        self.latest        = None                       #latest: stores last detected gate center & area.
        self.frame         = None                       #frame: stores last OpenCV frame for drawing & measurement.
        self.stop_detected = False
        self.stop_area     = 0.0
        self.target_gate = None    # lock-on center once we pick a gate

        # ALIGN gains + deadbands
        self.Kp_x, self.Kp_y = 0.3, 0.3                 #Kp_x, Kp_y: proportional gains for yaw (horizontal) and z (vertical) control in ALIGN.
        self.dead_x, self.dead_y = 0.2, 0.2             #dead_x, dead_y: small thresholds below which we consider “centered” and stop jitter.

        self.red_Kp_x, self.red_Kp_y = 0.3, 0.3                 #Kp_x, Kp_y: proportional gains for yaw (horizontal) and z (vertical) control in ALIGN.
        self.red_dead_x, self.red_dead_y = 0.2, 0.2            #dead_x, dead_y: small thresholds below which we consider “centered” and stop jitter.

        # timing & counters
        self.search_yaw_speed = -0.1                     #search_yaw_speed: how fast to spin in SEARCH.
        # self.forward_duration = 3.5                    #forward_duration: how many seconds to fly forward once aligned (guarantees passing).
        self.centered_frames  = 1                       #centered_frames: how many successive ALIGN frames must be “centered” before FORWARD. 
        self.centered_count   = 0                       #centered_count: counter for that.
        self.forward_start    = None                    #forward_start: timestamp when we began FORWARD.
        self.forward_duration   = 1.0 #may be lesser value like 0.2 is better but if it is toow slow of
        # self.fiducial_forward_duration = 2.5
        self.forward_flag = 0
        self.red_flag = 0
        self.keywords = [ "STO", "STOP", "TOP"]
        # self.stop_forward_duration = 4.0    # how many seconds to bump into the stop‐sign
        self.stop_forward_dur   = 0.25
        # # green HSV range
        # self.green_low  = np.array((35,  80,  80))
        # self.green_high = np.array((85, 255, 255))

        self.green_low  = np.array((50, 100,  10))        #green_low: lower bound of green HSV range.
        self.green_high = np.array((90, 220, 140))

        # self.l1,self.h1 = np.array((0,50,50)),   np.array((10,255,255))
        # self.l2,self.h2 = np.array((170,50,50)), np.array((180,255,255))
        self.l1,self.h1 = np.array((0,120,120)),   np.array((10,255,255))
        self.l2,self.h2 = np.array((170,120,120)), np.array((180,255,255))

    #         l1,h1 = (0,120,120),  (10,255,255)
    # l2,h2 = (170,120,120),(180,255,255)

        #Aruco marker detection
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.aruco_detector     = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # subscribe & publisher , Image subscription
        #We subscribe to the drone’s camera image topic /drone1/image_raw and publish velocity commands to /drone1/cmd_vel.
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  #Best‐effort because we don’t need every frame, just as many as possible at high rate.
            history=   HistoryPolicy.KEEP_LAST,
            depth=     10
        )
        self.create_subscription(Image,
                                 '/image_raw',
                                 self.image_cb,          #We call self.image_cb on each new image.
                                 qos_profile=camera_qos)
        
        self.annotated_image_pub = self.create_publisher(Image, '/gate_image_annotated', 10)
        
        #velocity command publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)  #Publisher to send velocity commands (Twist) to the drone.

        # takeoff/land service client
        self.cli = self.create_client(TelloAction, '/tello_action')  #We create a ROS 2 service client for the Tello “action” interface.
        while not self.cli.wait_for_service(timeout_sec=1.0):               #We block until the service is available.
            self.get_logger().info('waiting for /tello_action...')   
        self.takeoff()                                                      #Then we call our helper self.takeoff() to launch the drone.

        # main loop
        self.create_timer(1/20, self.control_loop)


    def takeoff(self):
        req = TelloAction.Request(); req.cmd='takeoff'                       #Builds a TelloAction request with cmd='takeoff'.
        self.cli.call_async(req); time.sleep(5.0)                            #Sends it asynchronously and then sleep(5 s) to let it climb
        time.sleep(10)
        cmd_publish = Twist()
        cmd_publish.linear.z = 0.4
        self.cmd_pub.publish(cmd_publish)
        time.sleep(2.0)
        cmd_publish.linear.z = 0.0
        self.cmd_pub.publish(cmd_publish)

        
    def land(self):
        req = TelloAction.Request(); req.cmd='land'                          #Similar for landing. Called at shutdown.
        self.cli.call_async(req)

    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image = frame

        gate = (self.detect_fiducial(frame))

        if gate is None:
            gate = self.detect_hole(frame) or self.detect_fiducial(frame)

        if gate:
            self.latest = gate
        else:
            stop = self.detect_stop(frame)
            if stop:
                self.latest = stop
                self.state  = 'ALIGN_STOP'
            else:
                self.latest = None

        # show & store
        # self.image = frame
        self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
        cv2.imshow('Gate Detector', self.image)                                  #Display the debug window.
        cv2.waitKey(1)
        # self.get_logger().info(self.latest and 'aligned' or 'no gate')      #Log whether we have a detection this frame or not.


    def detect_stop(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Optional: threshold to improve text clarity
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # # Use pytesseract to extract text
        # text = pytesseract.image_to_string(thresh) 
        # if any(k in text.upper() for k in self.keywords):
        #     print("STOP letters detected outside the red area")      
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        l1,h1 = np.array(self.l1), np.array(self.h1)
        l2,h2 = np.array(self.l2), np.array(self.h2)
        m = cv2.bitwise_or(cv2.inRange(hsv,l1,h1), cv2.inRange(hsv,l2,h2))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)

        cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            A = cv2.contourArea(c)
            print("Red:",A)
            # if any(k in text.upper() for k in self.keywords):
            #     print("STOP letters detected inside the red area")
            if A<300: 
                continue
            if A> 10000:
                self.land(); rclpy.shutdown()
            elif A> 800 :
                M = cv2.moments(c)
                if M['m00']==0: continue
                cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                cv2.drawContours(frame,[c],-1,(0,0,255),2)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                self.image = frame
                return {'center':(cx,cy),'area':A,'shape':'stop'}
        return None

    def detect_hole(self, frame):                                     
        """
        If no circle was found, try the old hole‐inside logic for squares.
        """
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)                     #Convert BGR → HSV.
        mask  = cv2.inRange(hsv, self.green_low, self.green_high)          #Threshold to green only.
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))               #A small 5×5 rectangle of “on” pixels (all ones) that we’ll slide over the image.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k1, iterations=1)    #Removes small white speckles (noise) that aren’t big enough to survive a 5×5 erosion.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=2)    #Fills small black holes inside the detected green area—important when the ring’s paint or rendering has tiny breaks.

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
            self.image = frame

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

        if not holes:                                                      #If no holes were found:
            return None
        
        best = None
        # prefer circle holes, else square
        for area,shape,c in holes:
            if shape=='circle':
                best = (area,shape,c); break                                #Among all holes, we pick the first circle (if any), else the largest square.
            if not best:
                best = (area,shape,c)

        if not best:
            return None

        area,shape,c = max(holes,key=lambda x:x[0])                        #We take the largest hole (area) and its shape.
        M = cv2.moments(c)                                                  #Compute the centroid of that hole via image moments.
        if M['m00']==0:
            return None
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        # draw fallback overlay
        color = (0,0,255) if shape=='circle' else (255,0,0)
        cv2.drawContours(frame,[c],-1,color,2)                              #Draw it in the debug window (red for circle, blue for square).
        cv2.circle(frame,(cx,cy),4,color,-1)
        self.image = frame
        return {'center':(cx,cy+250),'area':area}                               #Return the same kind of {'center':..., 'area':...} dict.


    def detect_fiducial(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None or len(ids)==0:
            return None
        if len(ids) ==3:
            centers = []
            for sq in corners:
                pts = sq[0]
                cx = pts[:,0].mean(); cy = pts[:,1].mean()
                centers.append((cx,cy))
                cv2.polylines(frame,[pts.astype(int)],True,(0,128,255),2)

            pts = np.array(centers,dtype=np.float32).reshape(-1,1,2).astype(np.int32)
            hull= cv2.convexHull(pts)
            M = cv2.moments(hull)
            if M['m00']==0: return None
            cx = int(M['m10']/M['m00']); cy=int(M['m01']/M['m00'])
            cv2.polylines(frame,[hull],True,(0,255,0),2)
            cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
            self.image = frame
            return {'center':(cx,cy+250),'area':cv2.contourArea(hull),'shape':'fiducial'}

    def control_loop(self):                                                     #Called at 20 Hz (we set that timer later). Drives the drone.

        if self.state == 'ALIGN_STOP':
            if self.latest:
                cx, cy = self.latest['center']
                h, w = self.frame.shape[:2]
                err_x = (cx - w/2) / (w/2)
                err_y = (h/2 - cy) / (h/2)

                cmd = Twist()
                if abs(err_x) > self.red_dead_x: cmd.angular.z = -self.red_Kp_x * err_x
                if abs(err_y) > self.red_dead_y: cmd.linear.z  =  self.red_Kp_y * err_y
                if abs(err_x) <= self.red_dead_x and abs(err_y) <= self.red_dead_y:
                # self.centered_frames    = 3
                    print("STOP centered. Moving forward.")
                    self.state = 'FORWARD_STOP'
                    self.forward_start = time.time()
                else:
                    self.cmd_pub.publish(cmd)
                    return

        if self.state=='FORWARD_STOP':
            if time.time()-self.forward_start < self.stop_forward_dur:
                cmd=Twist(); cmd.linear.x=0.2
                self.cmd_pub.publish(cmd)
                self.state='ALIGN_STOP'
            else:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)

            return
       
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

        if self.state == 'FORWARD':
            # If we've already passed through three gates, go long
            if self.forward_flag == 2:
                self.forward_flag = 0
                print("forward reset")
                f = Twist()
                f.linear.x = 0.4
                self.cmd_pub.publish(f)
                time.sleep(3.5)
                # stop and go back to SEARCH
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)
                self.target_gate = None
                self.state = 'SEARCH'    
                return

            # Otherwise, short forward bursts
            if self.forward_flag < 2:
                #self.forward_duration = 0.5
                self.forward_flag += 1
                # self.forward_start = time.time() 
                print("forward incremented")
                f = Twist()
                f.linear.x = 0.2
                self.cmd_pub.publish(f)
                time.sleep(0.7)
                # stop and go back to SEARCH
                self.cmd_pub.publish(Twist())
                time.sleep(0.1)  
                self.target_gate = None
                self.state = 'SEARCH'
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



