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


def detect_ring(img_bgr):
    blurred = cv2.GaussianBlur(img_bgr, (5,5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # start with very loose bounds
    lower = np.array([40,  50,  50])
    upper = np.array([80, 255, 255])
    mask  = cv2.inRange(hsv, lower, upper)

    # show the mask window
    cv2.imshow('debug mask', mask)

    cnts, _ = cv2.findContours(mask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    gate = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(gate)
    if area < 2000:
        return None

    x,y,w,h = cv2.boundingRect(gate)
    cx,cy   = x + w//2, y + h//2

    # print average HSV inside that bbox
    roi = hsv[y:y+h, x:x+w]
    h_mean, s_mean, v_mean = cv2.mean(roi)[:3]
    print(f"[HSV] {h_mean:.0f}, {s_mean:.0f}, {v_mean:.0f}")

    return {'bbox':(x,y,w,h),'center':(cx,cy),'area':area}

    pass

class GateFlyerSim(Node):
    def __init__(self):
        super().__init__('gate_flyer_sim')
        self.bridge = CvBridge()
        self.state  = 'SEARCH'
        self.latest = None
        self.frame  = None

        # PD controller gains + deadbands
        self.Kp_x, self.Kd_x = 0.6, 0.1
        self.Kp_y, self.Kd_y = 0.4, 0.05
        self.dead_x, self.dead_y = 0.05, 0.05

        # keep history for derivative
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time  = self.get_clock().now()

        # subscribe with BEST_EFFORT QoS
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=   HistoryPolicy.KEEP_LAST,
            depth=     5
        )
        self.create_subscription(Image,
                                 '/drone1/image_raw',
                                 self.image_cb,
                                 qos_profile=camera_qos)
        self.cmd_pub = self.create_publisher(Twist,
                                             '/drone1/cmd_vel', 10)

        # takeoff/land client
        self.cli = self.create_client(TelloAction, '/drone1/tello_action')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for /drone1/tello_action...')
        self.takeoff()

        # control loop at 20 Hz
        self.create_timer(1/20, self.control_loop)

    def takeoff(self):
        req = TelloAction.Request(); req.cmd = 'takeoff'
        self.cli.call_async(req); time.sleep(5.0)

    def land(self):
        req = TelloAction.Request(); req.cmd = 'land'
        self.cli.call_async(req)

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gate = detect_ring(img)
        self.frame  = img
        self.latest = gate

        # debug overlay
        if gate:
            x,y,w,h = gate['bbox']; cx,cy = gate['center']
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(img,(cx,cy),4,(0,0,255),-1)
        cv2.imshow('GateDetectorSim',img)
        cv2.waitKey(1)

    def compute_pd_twist(self, centre, shape, area):
        h,w = shape[:2]
        err_x = (centre[0] - w/2) / (w/2)
        err_y = (h/2 - centre[1]) / (h/2)

        now    = self.get_clock().now()
        dt      = (now - self.prev_time).nanoseconds * 1e-9
        d_err_x = (err_x - self.prev_err_x)/dt if dt>1e-6 else 0.0
        d_err_y = (err_y - self.prev_err_y)/dt if dt>1e-6 else 0.0

        # PD outputs
        yaw =  self.Kp_x*err_x + self.Kd_x*d_err_x
        z   =  self.Kp_y*err_y + self.Kd_y*d_err_y

        # deadband small jitters
        if abs(err_x) < self.dead_x: yaw = 0.0
        if abs(err_y) < self.dead_y: z   = 0.0

        # clamp to safe ranges
        yaw = max(min(yaw, 0.5), -0.5)
        z   = max(min(z,   0.3), -0.3)

        # forward speed (unchanged P‐law)
        tgt_area = 20000.0
        fwd = 0.4 * max(0.0, 1 - min(area/tgt_area,1.0))

        # build Twist
        cmd = Twist()
        cmd.angular.z = yaw
        cmd.linear.z  = z
        cmd.linear.x  = fwd

        # update history
        self.prev_err_x = err_x
        self.prev_err_y = err_y
        self.prev_time  = now

        return cmd

    def control_loop(self):
        if self.state == 'SEARCH':
            if self.latest:
                self.state = 'ALIGN'
            else:
                t=Twist(); t.angular.z=0.2; self.cmd_pub.publish(t)
            return

        if self.state == 'ALIGN':
            if not self.latest:
                self.state='SEARCH'; return
            cmd = self.compute_pd_twist(
                     self.latest['center'],
                     self.frame.shape,
                     self.latest['area'])
            self.cmd_pub.publish(cmd)
            # once nearly centered & close → go THROUGH
            if (abs(cmd.angular.z)<0.05 and
                abs(cmd.linear.z)<0.05 and
                self.latest['area']>15000):
                self.state='THROUGH'
            return

        if self.state == 'THROUGH':
            t=Twist(); t.linear.x=0.5; self.cmd_pub.publish(t)
            if not self.latest:
                self.state='SEARCH'
            return

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = GateFlyerSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.land()
        rclpy.shutdown()

if __name__=='__main__':
    main()





# #!/usr/bin/env python3
# import time
# import rclpy
# import cv2
# import numpy as np

# from rclpy.node        import Node
# from rclpy.qos         import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from sensor_msgs.msg   import Image
# from geometry_msgs.msg import Twist
# from tello_msgs.srv    import TelloAction
# from cv_bridge         import CvBridge

# # def detect_ring(img_bgr):
# #     """
# #     Detect the ring by green‐border color segmentation.
# #     Returns dict with bbox, center, area or None if nothing found.
# #     """
# #     blurred = cv2.GaussianBlur(img_bgr, (5,5), 0)
# #     hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# #     # tweak these to match your ring's color
# #     lower = np.array([50, 120, 120])
# #     upper = np.array([80, 255, 255])
# #     mask  = cv2.inRange(hsv, lower, upper)

# #     # clean up
# #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# #     mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# #     cnts, _ = cv2.findContours(mask,
# #                                cv2.RETR_EXTERNAL,
# #                                cv2.CHAIN_APPROX_SIMPLE)
# #     if not cnts:
# #         return None

# #     gate = max(cnts, key=cv2.contourArea)
# #     area = cv2.contourArea(gate)
# #     if area < 2000:  # noise floor
# #         return None

# #     x,y,w,h = cv2.boundingRect(gate)
# #     cx,cy   = x + w//2, y + h//2
# #     return {'bbox':(x,y,w,h), 'center':(cx,cy), 'area':area}

# def detect_ring(img_bgr):
#     blurred = cv2.GaussianBlur(img_bgr, (5,5), 0)
#     hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#     # start with very loose bounds
#     lower = np.array([40,  50,  50])
#     upper = np.array([80, 255, 255])
#     mask  = cv2.inRange(hsv, lower, upper)

#     # show the mask window
#     cv2.imshow('debug mask', mask)

#     cnts, _ = cv2.findContours(mask,
#                                cv2.RETR_EXTERNAL,
#                                cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None

#     gate = max(cnts, key=cv2.contourArea)
#     area = cv2.contourArea(gate)
#     if area < 2000:
#         return None

#     x,y,w,h = cv2.boundingRect(gate)
#     cx,cy   = x + w//2, y + h//2

#     # print average HSV inside that bbox
#     roi = hsv[y:y+h, x:x+w]
#     h_mean, s_mean, v_mean = cv2.mean(roi)[:3]
#     print(f"[HSV] {h_mean:.0f}, {s_mean:.0f}, {v_mean:.0f}")

#     return {'bbox':(x,y,w,h),'center':(cx,cy),'area':area}

# def compute_twist(center, frame_shape, area):
#     """
#     P‐controller to center ring in frame and move forward.
#     """
#     h, w = frame_shape[:2]
#     err_x = (center[0] - w/2) / (w/2)   # yaw error [-1..1]
#     err_y = (h/2 - center[1])  / (h/2)  # altitude error [-1..1]

#     cmd = Twist()
#     cmd.angular.z = -0.5 * err_x       # yaw left/right
#     cmd.linear.z  =  0.3 * err_y       # up/down

#     # slower forward as we near the ring
#     target_area = 20000.0
#     forward = max(0.0, 1 - min(area/target_area,1.0))
#     cmd.linear.x = 0.4 * forward
#     return cmd

# class GateFlyerSim(Node):
#     def __init__(self):
#         super().__init__('gate_flyer_sim')
#         self.bridge = CvBridge()
#         self.state  = 'SEARCH'
#         self.latest = None
#         self.frame  = None

#         # --- BEST_EFFORT QoS for Gazebo camera ---
#         camera_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=   HistoryPolicy.KEEP_LAST,
#             depth=     5
#         )
#         self.create_subscription(
#             Image,
#             '/drone1/image_raw',
#             self.image_cb,
#             qos_profile=camera_qos
#         )

#         self.cmd_pub = self.create_publisher(
#             Twist,
#             '/drone1/cmd_vel',
#             10
#         )

#         # service for takeoff/land
#         self.cli = self.create_client(TelloAction, '/drone1/tello_action')
#         while not self.cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('waiting for /drone1/tello_action...')
#         self.takeoff()

#         # run loop at 20 Hz
#         self.create_timer(1/20, self.control_loop)

#     def takeoff(self):
#         req = TelloAction.Request()
#         req.cmd = 'takeoff'
#         self.cli.call_async(req)
#         time.sleep(5.0)

#     def land(self):
#         req = TelloAction.Request()
#         req.cmd = 'land'
#         self.cli.call_async(req)

#     def image_cb(self, msg: Image):
#         img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
#         gate = detect_ring(img)
#         self.frame  = img
#         self.latest = gate

#         # draw overlay
#         if gate:
#             x,y,w,h = gate['bbox']
#             cx,cy   = gate['center']
#             cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
#             cv2.circle(img, (cx,cy),4,(0,0,255),-1)
#         cv2.imshow('GateDetectorSim', img)
#         cv2.waitKey(1)

#     def control_loop(self):
#         if self.state == 'SEARCH':
#             if self.latest:
#                 self.state = 'ALIGN'
#             else:
#                 t = Twist(); t.angular.z = 0.05
#                 self.cmd_pub.publish(t)
#             return

#         if self.state == 'ALIGN':
#             if not self.latest:
#                 self.state = 'SEARCH'
#                 return
#             cmd = compute_twist(self.latest['center'],
#                                 self.frame.shape,
#                                 self.latest['area'])
#             self.cmd_pub.publish(cmd)
#             # if roughly centered and near
#             if (abs(cmd.angular.z)<0.05 and
#                 abs(cmd.linear.z)<0.05 and
#                 self.latest['area']>15000):
#                 self.state = 'THROUGH'
#             return

#         if self.state == 'THROUGH':
#             # punch through
#             t = Twist(); t.linear.x=0.5
#             self.cmd_pub.publish(t)
#             # once we stop seeing the ring assume we passed
#             if not self.latest:
#                 self.state = 'SEARCH'
#             return

#     def destroy_node(self):
#         cv2.destroyAllWindows()
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = GateFlyerSim()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.land()
#         rclpy.shutdown()

# if __name__=='__main__':
#     main()
