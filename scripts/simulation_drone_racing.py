#!/usr/bin/env python3

# ──────────────────────────────────────────────────────────────────────────────  
# ──────────────────────────────────────────────────────────────────────────────

#command line usage:
#python3 simulation_drone_racing.py            # ➔ real-world mode
#python3 simulation_drone_racing.py --sim      # ➔ simulation mode

# ──────────────────────────────────────────────────────────────────────────────

import sys
import time
import rclpy
import cv2
import numpy as np
import math

from rclpy.node        import Node
from rclpy.qos         import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg   import Image
from geometry_msgs.msg import Twist
from tello_msgs.srv    import TelloAction
from cv_bridge         import CvBridge
import cv2.aruco as aruco
import pytesseract

class DroneRacing(Node):
    def __init__(self, simulation=False):
        super().__init__('shape_flyer')
        self.bridge = CvBridge()
        self.simulation = simulation
        # ─── FSM state ───
        self.state        = 'SEARCH'   # start by climbing
        self.latest       = None      # last detected object (gate or stop)
        self.frame        = None

        # ─── ALIGN gains + deadbands ───
        if self.simulation: #simulated drone
            self.Kp_x, self.Kp_y = 0.1, 0.1                 #Kp_x, Kp_y: proportional gains for yaw (horizontal) and z (vertical) control in ALIGN.
            self.dead_x, self.dead_y = 0.1, 0.1

        else: #real drone
            self.Kp_x, self.Kp_y = 0.3, 0.3
            self.dead_x, self.dead_y = 0.2, 0.2

        # ─── timing & counters ───
        if self.simulation:
            self.search_yaw_speed   = 0.3
            self.centered_frames    = 3
            self.centered_count     = 0
            self.forward_duration   = 0.25
            self.forward_start      = None
            self.stop_forward_dur   = 1.0
        else:
            self.search_yaw_speed   = 0.1
            self.centered_frames    = 1
            self.centered_count     = 0
            self.forward_duration   = 0.5 #may be lesser value like 0.2 is better but if it is toow slow of
            self.fiducial_forward_duration = 3.0
            self.forward_flag = 0
            #ddoesnt detect change ths to 0.5
            self.forward_start      = None
            self.stop_forward_dur   = 0.25


        self.keywords = ["S", "ST", "STO", "STOP", "T", "O", "P"]
        self.gates = []
        self.image = None
        self.fiducial = 0


        #Gate Green HSV range
        if self.simulation:
            self.green_low  = np.array((35,  80,  80))
            self.green_high = np.array((85, 255, 255))
        else:
            # self.green_low  = np.array((35, 140,  0))        #green_low: lower bound of green HSV range.
            # self.green_high = np.array((85, 230, 255))
            self.green_low  = np.array((50, 100,  10))        #green_low: lower bound of green HSV range.
            self.green_high = np.array((90, 220, 140))

        #red HSV range
        if self.simulation:
            self.stop_low  = np.array((0,  50,  0))        #red_low: lower bound of green HSV range.
            self.stop_high = np.array((180, 255, 50))
            self.l1 = (0, 50, 0)
            self.h1 = (10, 255, 80)
            self.l2 = (170, 50, 0)
            self.h2 = (180, 255, 80)
        else:
            self.stop_low  = np.array((0, 180,  110))        #red_low: lower bound of green HSV range.
            self.stop_high = np.array((10, 215, 255))
            self.l1 = (0,100,80)
            self.h1 = (5,255,255)
            self.l2 = (175,100,80)
            self.h2 = (180,255,255)


        # ─── BLIND_FORWARD ───
        if self.simulation: 
            self.blind_duration   = 0.7
            self.blind_flag = 0
            
        else:
            self.blind_duration   = 3.0
            self.blind_flag = 0

        self.blind_start      = None
        # ─── ArUco setup ───
        self.aruco_dict      = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params    = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # ─── ROS interfaces ───
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        if self.simulation: #simulated drone
            print("Simulation mode")
            self.create_subscription(
                Image, '/drone1/image_raw', self.image_cb, qos_profile=camera_qos
            )
            self.annotated_image_pub = self.create_publisher(Image, '/drone1/gate_image_annotated', 10)
            self.cmd_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
            self.cli     = self.create_client(TelloAction, '/drone1/tello_action')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('waiting for /drone1/tello_action…')
            self.takeoff()

        else: #real drone
            print("Real drone mode")
            self.create_subscription(
                Image, '/image_raw', self.image_cb, qos_profile=camera_qos
            )
            self.annotated_image_pub = self.create_publisher(Image, '/gate_image_annotated', 10)
            self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
            self.cli     = self.create_client(TelloAction, '/tello_action')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('waiting for /tello_action…')
            self.takeoff()

        # ─── control loop @30 Hz ───
        self.create_timer(1/30, self.control_loop)

    
    
    def takeoff(self):
        req = TelloAction.Request(); req.cmd='takeoff'
        self.cli.call_async(req)
        time.sleep(5.0)
        if self.simulation:
            return
        # if real drone, wait for takeoff to finish
        else:
            time.sleep(10)
            cmd_publish = Twist()
            cmd_publish.linear.z = 0.4
            self.cmd_pub.publish(cmd_publish)
            time.sleep(2.0)
            cmd_publish.linear.z = 0.0
            self.cmd_pub.publish(cmd_publish)


    def land(self):
        req = TelloAction.Request(); req.cmd='land'
        self.cli.call_async(req)



    def image_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.image = frame.copy()
        self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
        fiducial = self.detect_fiducial(frame)
        if fiducial:
            # print("Fiducial detected")
            self.latest = fiducial
            self.fiducial = 1
            return
        # Priority 1: Try detecting hole
        hole = self.detect_hole(frame)
        if hole:
            # print("Hole detected")
            self.latest = hole
            self.fiducial = 0
            return


        # Priority 4: If no gates detected, try detecting STOP sign
        stop = self.detect_stop(frame)
        
        if stop:
            self.fiducial = 0
            print("STOP sign detected")
            self.latest = stop
            self.state = 'ALIGN_STOP'
            return

        # Nothing detected
        self.latest = None

    
    def detect_stop(self, frame):
        # if self.blind_flag >3:
        print("blind flag= ", self.blind_flag)
        if self.simulation:
            # Convert to grayscale (helps OCR accuracy)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optional: threshold to improve text clarity
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Use pytesseract to extract text
            text = pytesseract.image_to_string(thresh) 
            if any(k in text.upper() for k in self.keywords):
                print("STOP letters detected outside the red area")      
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
                if any(k in text.upper() for k in self.keywords):
                    print("STOP letters detected inside the red area")
                if A<500: 
                    continue
                if A> 45000:
                    self.land(); rclpy.shutdown()
                elif A> 6000:
                    M = cv2.moments(c)
                    if M['m00']==0: continue
                    cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                    cv2.drawContours(frame,[c],-1,(0,0,255),2)
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    self.image = frame
                    self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
                    return {'center':(cx,cy),'area':A,'shape':'stop'}
            return None
        else:
                    # Convert to grayscale (helps OCR accuracy)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optional: threshold to improve text clarity
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            # Use pytesseract to extract text
            text = pytesseract.image_to_string(thresh) 
            if any(k in text.upper() for k in self.keywords):
                print("STOP letters detected outside the red area")      
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
                if A> 5000 :
                    self.land(); rclpy.shutdown()
                elif A> 500 :
                    M = cv2.moments(c)
                    if M['m00']==0: continue
                    cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                    cv2.drawContours(frame,[c],-1,(0,0,255),2)
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    self.image = frame
                    self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
                    cv2.imshow('Drone View', self.image)
                    cv2.waitKey(1)
                    return {'center':(cx,cy),'area':A,'shape':'stop'}
            return None


    def edge_detector(self, image):
        if len(image.shape) == 3:
            # Convert the image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize the image
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # Equalize the histogram of the image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        # Blur the image to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        # Detecte edges with laplacian of gaussian
        image = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        # Convert the image to absolute values
        image = cv2.convertScaleAbs(image)
        image = cv2.addWeighted(image, 1.5, image, 0, 0)
        # Apply median blur to reduce noise
        image = cv2.medianBlur(image, 3)
        # Apply Otsu's thresholding
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -7)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        image = cv2.morphologyEx(image , cv2.MORPH_CLOSE, kernel, iterations=1)
        return image
    

    def gate_detector(self, image):
        # Create a copy of the image
        image = image.copy()
        image = self.edge_detector(image)
        # Calculate the contours of the image 
        contours, hierarchies = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort the contours based on the area of the bounding box
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:10]
        # Reconvert the image to display the contours with color
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Find the gate
        self.gates = []
        for contour in contours:
            cv2.drawContours(self.image, [contour], 0, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate the area of the gate
            # area_contour = cv2.contourArea(contour)
            area = cv2.contourArea(contour)
            area_contour = area
            area = area / (image.shape[0] * image.shape[1])
            # Calculate the ratio of the bounding box to see if it is a square
            # ratio = w / h
            rect_area = w * h
            # print("Ratio: ", ratio)
            peri = cv2.arcLength(contour, True)
            if rect_area ==0 or peri == 0:
                continue
# if 0.85 < ratio < 1.2 and 0.010 < area < 0.55 and solidity > 0.9 and cy < h_img * 0.75:
    # Calculate the center of the gate

            extent = area_contour / rect_area
            circularity = 4 * np.pi * area_contour / (peri * peri)
            # Ratio:{ratio:.2f}, 

            print(f"area : {area} ,Extent: {extent:.2f} Circularity: {circularity:.2f}")
            # Draw the bounding box
            # If the ratio is a square and the area is between 2% and 60% of the image
            # 0.85 < ratio < 1.4 and
            # if (area )
            # and (extent > 0.7) 
            if ( 0.015 <= area <= 0.75
                and (extent >=0.5)
                and (circularity >= 0.5)) :
                # Calculate the center of the gate
                cx = x + w / 2
                cy = y + h / 2
                # Save the gate
                self.gates.append((x, y, w, h, int(cx), int(cy), np.round(area, 2)))

        # Sort the gates based on the area of the bounding box
        self.gates = sorted(self.gates, key=lambda x: x[6], reverse=True)
        # print(self.gates)
        # if self.gates[0] is not None:
        if len(self.gates) > 0:
            x, y, w, h, cx, cy, area = self.gates[0]
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(self.image, (cx, cy), 5, (0, 0, 255), -1)
            # cv2.putText(self.image, "Curr", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.image, "Area: {:.2f}".format(area), (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
        cv2.imshow('Drone View', self.image)
        cv2.waitKey(1)
        return image
    





    def detect_hole(self, frame):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_low, self.green_high)
        k1   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k1,iterations=2) 
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,k1,iterations=2)
        mg      = cv2.bitwise_and(hsv, hsv, mask=mask)
        image = cv2.cvtColor(mg, cv2.COLOR_HSV2BGR)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        #end of background removal
        image_gates=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gates = cv2.adaptiveThreshold(image_gates, 255, 
                                            cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY, 5, 2)
        self.frame = self.gate_detector(image_gates)      
        if len(self.gates) > 0:
            # print(self.gates[0])
            x, y, w, h, cx, cy, A = self.gates[0]
            if self.simulation:
                return {'center':(cx,cy),'area':A,'shape':'circle'}
            else:
                return {'center':(cx,cy+150),'area':A,'shape':'frame'}
        else:
            return None 


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
            self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, 'bgr8'))
            cv2.imshow('Drone View', self.image)
            cv2.waitKey(1)
            if self.simulation:
                return {'center':(cx,cy),'area':cv2.contourArea(hull),'shape':'fiducial'}
            else:
                return {'center':(cx,cy+100),'area':cv2.contourArea(hull),'shape':'fiducial'}


    def control_loop(self):
        if self.state == 'BLIND_FORWARD':
           
            # h,w=self.frame.shape[:2]
            if time.time() - self.blind_start < self.blind_duration:
                cmd = Twist(); cmd.linear.x = 0.4
                self.cmd_pub.publish(cmd)
            else:
                self.cmd_pub.publish(Twist())
                self.state = 'SEARCH'
            return        


        if self.state == 'ALIGN_STOP':
            if self.latest:
                cx, cy = self.latest['center']
                h, w = self.frame.shape[:2]
                ex = (cx - w/2) / (w/2)
                ey = (h/2 - cy) / (h/2)

                cmd = Twist()

                # if abs(ex) > self.dead_x:
                #     cmd.angular.z = -self.Kp_x * ex  # signed rotation

                # if abs(ey) > self.dead_y:
                #     cmd.linear.z = self.Kp_y * ey  # signed up/down
                if abs(ex) > self.dead_x:
                    cmd.angular.z = -0.3 * np.tanh(2.5 * ex)

                if abs(ey) > self.dead_y:
                    cmd.linear.z = 0.2 * np.tanh(2.5 * ey)
                # ✅ Clip maximum values
                    cmd.angular.z = np.clip(cmd.angular.z, -0.3, 0.3)
                    self.search_yaw_speed   = 0.3
                    self.centered_frames    = 3
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

        # GATE SEARCH / ALIGN / FORWARD
        if self.state=='SEARCH':
            if self.latest:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                self.state='ALIGN'; self.centered_count=0
            else:
                cmd=Twist(); cmd.angular.z=0.1
                self.cmd_pub.publish(cmd)
            return
 
        if self.state=='ALIGN':
            if not self.latest:
                self.state='SEARCH'; return
            cx,cy=self.latest['center']
            h,w=self.frame.shape[:2]
            ex=(cx-w/2)/(w/2); ey=(h/2-cy)/(h/2)
            cmd=Twist()
            if abs(ex) > self.dead_x:
                cmd.angular.z = -self.Kp_x * ex       # yaw toward gate

            # # 2.   Only use sideways translation when yaw is nearly zero
            # yaw_aligned = abs(ex) < 0.15             # tune 0.10‑0.20
            # if yaw_aligned:
            #     cmd.linear.y = 0.1 * ex             # roll toward gate
            #     #      ^^^^^  negative sign → right drift = ‑y
            #     # 0.2 m/s is safe for Tello; clamp later

            # 3.   Vertical alignment (unchanged)
            if abs(ey) > self.dead_y:
                cmd.linear.z =  self.Kp_y * ey

            # 4.   Clip all velocities
            cmd.angular.z = np.clip(cmd.angular.z, -0.4, 0.4)
            cmd.linear.y  = np.clip(cmd.linear.y,  -0.3, 0.3)
            cmd.linear.z  = np.clip(cmd.linear.z,  -0.3, 0.3)
            if abs(ex)<=self.dead_x and abs(ey)<=self.dead_y:
                self.centered_count+=1
            else:
                self.centered_count=0
            if self.centered_count>=self.centered_frames:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                self.forward_start=time.time()
                if self.fiducial ==1:
                    self.state = "Fiducial_Forward"
                else:
                    self.state='FORWARD'

            else:
                self.cmd_pub.publish(cmd)
            return
        
        if self.state=='Fiducial_Forward':
            if not self.latest:
                self.state='SEARCH'; return
            print("Inside the fiducial forward")
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



        if self.state=='FORWARD':
            if not self.latest:
                self.state='SEARCH'; return

            h,w   = self.frame.shape[:2]
            if self.simulation:
                error = np.round(np.abs(0.30 - self.latest['area']), 2) 
            # print(self.latest['area'])
            else:
                error = np.round(np.abs(0.20 - self.latest['area']), 2)

            # and self.latest['area'] >= 0.5 : 
            # if gate fills >60% of frame, go BLIND_FORWARD
            if self.latest and  error<0.1 :  
                print('**************************BLIND_FORWARD**************************************')
                self.state      = 'BLIND_FORWARD'
                self.blind_flag = self.blind_flag +1
                self.blind_start = time.time()
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                return
            if time.time()-self.forward_start < self.forward_duration:
                cmd=Twist(); cmd.linear.x=0.2
                self.cmd_pub.publish(cmd)
            else:
                self.cmd_pub.publish(Twist()); time.sleep(0.1)
                self.state='SEARCH'
            return


    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

# ──────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    simulation = "--sim" in sys.argv
    node = DroneRacing(simulation=simulation)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()
        rclpy.shutdown()


if __name__=='__main__':
    main()









            # if self.fiducial == 1:
            #     print("Inside the fiducial forward")
            #     if self.forward_flag ==3:
            #         self.forward_start = time.time()
            #         if time.time()-self.forward_start < self.fiducial_forward_duration:
            #             cmd=Twist(); cmd.linear.x=0.4
            #             self.cmd_pub.publish(cmd)
            #             self.forward_flag =0
            #             print("forward reset")
            #         else:
            #             self.cmd_pub.publish(Twist()); time.sleep(0.1)
            #             self.state='SEARCH'
            #         return
            #     if self.forward_flag < 3:
            #         self.forward_start = time.time()
            #         if time.time()-self.forward_start < self.forward_duration:
            #             cmd=Twist(); cmd.linear.x=0.2
            #             self.cmd_pub.publish(cmd)
            #             self.forward_flag = self.forward_flag+1
            #             print("forward flag incremented")
            #         else:
            #             self.cmd_pub.publish(Twist()); time.sleep(0.1)
            #             self.state='SEARCH'
            #         return
