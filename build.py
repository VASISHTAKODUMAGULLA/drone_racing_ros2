

import time
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
from cv_bridge import CvBridge
from djitellopy import Tello

class TelloNode(Node):
    def __init__(self):
        super().__init__('tello')
        self.bridge = CvBridge()
        self.latest_image = None
        self.tello_env = 1 # if 1 it is in the real world if not in the simulation.

        # Create an OpenCV window in the main thread


        qos_policy = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.camera_callback,
            qos_profile=qos_policy
        )
        if self.tello_env == 1:
            print("real world")
            # self.cli = self.create_client(TelloAction, '/tello_action')
            # while not self.cli.wait_for_service(timeout_sec=1.0):
            #     self.get_logger().info('service not available, waiting again...')
            # self.req = TelloAction.Request()
            # self.get_logger().info("inside takeoff")
            # self.req.cmd = 'takeoff'
            # self.future = self.cli.call_async(self.req)
            # time.sleep(15)
            # self.req.cmd = 'land'
            # self.future = self.cli.call_async(self.req)
            # print("real world")
            # tello = Tello.connect()
            # tello.takeoff()
            # time.sleep(20)
            # tello.land()

        else:
            self.cli = self.create_client(TelloAction, '/drone1/tello_action')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('service not available, waiting again...')
            self.req = TelloAction.Request()
            self.get_logger().info("inside takeoff")
            self.req.cmd = 'takeoff'
            self.future = self.cli.call_async(self.req)
            time.sleep(100)
            self.req.cmd = 'land'
            self.future = self.cli.call_async(self.req)
            

        # You might want to delay sending velocity commands until after takeoff completes
        # self.create_timer(3.0, self.send_velocity_command)

        # Timer to update the OpenCV window at ~30Hz from the main thread
        # self.create_timer(0.03, self.timer_callback)

    def camera_callback(self, msg):
        self.get_logger().info("Received an image message.")

        # Convert ROS image message to OpenCV image (BGR)
        image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # self.get_logger().info(f"{type(image)}")
        cv2.imshow("ros2image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

            # self.get_logger().info("converted image")

            # cv2.imwrite('./image1.png', image)
            # self.latest_image = image
        # except Exception as e:
        #     self.get_logger().error(f"Failed to convert image: {e}")

    # def timer_callback(self):
    #     # This runs in the main thread and updates the window
    #     if self.latest_image is not None:
    #         print("image works")
    #         # display_image = cv.resize(self.latest_image, (720, 540), interpolation=cv.INTER_LINEAR)
    #         cv.imshow("image", self.latest_image)
    #         cv.waitKey(1)  # Process GUI events

    def send_velocity_command(self):
        cmd_publish = Twist()
        cmd_publish.linear.z = 0.1
        self.publisher_.publish(cmd_publish)

def main(args=None):
    rclpy.init(args=args)
    node = TelloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()






# import rclpy
# import cv2
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

# class DisplayNode(Node):
#     def __init__(self):
#         super().__init__('display_node')
#         self.bridge = CvBridge()
#         self.latest_image = None
#         cv2.namedWindow("image", cv2.WINDOW_NORMAL)
#         self.create_subscription(Image, '/drone1/image_raw', self.image_callback, 10)
#         # Timer to update the display at 30 Hz
#         self.create_timer(1.0 / 30.0, self.timer_callback)

#     def image_callback(self, msg):
#         try:
#             self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         except Exception as e:
#             self.get_logger().error(f"Image conversion error: {e}")

#     def timer_callback(self):
#         if self.latest_image is not None:
#             cv2.imshow("image", self.latest_image)
#             # Exit if 'q' is pressed (optional)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 rclpy.shutdown()

# def main(args=None):
#     rclpy.init(args=args)
#     node = DisplayNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Shutting down...")
#     finally:
#         cv2.destroyAllWindows()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()



























# import rclpy
# import time
# from rclpy.node import Node
# import numpy as np
# import cv2
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# import matplotlib.pyplot as plt
# import sys
# from tello_msgs.srv import TelloAction
# from djitellopy import Tello
# from cv_bridge import CvBridge

# # tello = Tello()

# class Tello(Node):
#     # global k
#     def __init__(self):
#         super().__init__('tello')
#         qos_policy = rclpy.qos.QoSProfile(
#             reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
#             history=rclpy.qos.HistoryPolicy.KEEP_LAST,
#             depth=1
#         )
#         self.publisher_ = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
#         self.subscription = self.create_subscription(
#             Image,
#             '/drone1/image_raw',
#             self.camera_callback,
#             qos_profile=qos_policy 
#         )
#         self.bridge = CvBridge()
#         # To Send the takeoff and land commands
#         self.cli = self.create_client(TelloAction, '/drone1/tello_action')
#         while not self.cli.wait_for_service(timeout_sec=1.0):
#             self.get_logger().info('service not available, waiting again...')
#         self.req = TelloAction.Request()
#         print("inside takeoff")
#         self.req.cmd = 'takeoff'
#         self.future = self.cli.call_async(self.req)
#         time.sleep(3)
#         cmd_publish = Twist()
#         cmd_publish.linear.z = 0.1
#         self.publisher_.publish(cmd_publish)
#         time.sleep(1)
#         cmd_publish.linear.z = 0.0
#         self.publisher_.publish(cmd_publish)
#         cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    
#     def camera_callback(self, msg):
#         print("Hi")
#         imageFrame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         imageFrame=cv2.resize(imageFrame, (780, 540),
#                interpolation = cv2.INTER_LINEAR)
#         # imageFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2RGB)
#         # hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_RGB2HSV)
#         # green_lower = np.array([25, 52, 72], np.uint8)
#         # green_upper = np.array([102, 255, 255], np.uint8)
#         # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

#         # red_lower = np.array([111, 87, 136], np.uint8)
#         # red_upper = np.array([180, 255, 255], np.uint8)
#         # red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

#         # blue_lower = np.array([94, 80, 2], np.uint8)
#         # blue_upper = np.array([120, 255, 255], np.uint8)
#         # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

#         # kernel = np.ones((5, 5), "uint8")

#         # green_mask = cv2.dilate(green_mask, kernel)
#         # res_green = cv2.bitwise_and(imageFrame, imageFrame,
#         #                             mask = green_mask)
#         # # For red color
#         # red_mask = cv2.dilate(red_mask, kernel)
#         # res_red = cv2.bitwise_and(imageFrame, imageFrame, 
#         #                         mask = red_mask)

#         # # For blue color
#         # blue_mask = cv2.dilate(blue_mask, kernel)
#         # res_blue = cv2.bitwise_and(imageFrame, imageFrame,
#         #                         mask = blue_mask) 

#         #  # Creating contour to track red color
#         # contours_red, hierarchy = cv2.findContours(red_mask,
#         #                                    cv2.RETR_TREE,
#         #                                    cv2.CHAIN_APPROX_SIMPLE)
      
#         # for pic, contour in enumerate(contours_red):
#         #     area = cv2.contourArea(contour)
#         #     if(area > 300):
#         #         x, y, w, h = cv2.boundingRect(contour)
#         #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
#         #                                 (x + w, y + h), 
#         #                                 (0, 0, 255), 2)

#         #  # Creating contour to track green color
#         # contours_green, hierarchy = cv2.findContours(green_mask,
#         #                                    cv2.RETR_TREE,
#         #                                    cv2.CHAIN_APPROX_SIMPLE)
      
#         # for pic, contour in enumerate(contours_green):
#         #     area = cv2.contourArea(contour)
#         #     if(area > 300):
#         #         x, y, w, h = cv2.boundingRect(contour)
#         #         imageFrame = cv2.rectangle(imageFrame, (x, y), 
#         #                                 (x + w, y + h),
#         #                                 (0, 255, 0), 2)

#         # # Creating contour to track blue color
#         # contours_blue, hierarchy = cv2.findContours(blue_mask,
#         #                                     cv2.RETR_TREE,
#         #                                     cv2.CHAIN_APPROX_SIMPLE)
#         # for pic, contour in enumerate(contours_blue):
#         #     area = cv2.contourArea(contour)
#         #     if(area > 300):
#         #         x, y, w, h = cv2.boundingRect(contour)
#         #         imageFrame = cv2.rectangle(imageFrame, (x, y),
#         #                                 (x + w, y + h),
#         #                                 (255, 0, 0), 2)

#         # print(len(contours_green), "contours")
#         # shape = self.shape_detection(res_green)
#         # print(shape,"shape")
#         # if (len(contours_green) and (shape==1 or shape ==2)):

#         #     print("detected")
#         #         # area = cv2.contourArea(contour)
#         #         # cntrs = list(contours)
#         #     # list for storing names of shapes
#         #         # cntours = []
#         #         # for pos,contour in enumerate(contours_green):
#         #         #     area = cv2.contourArea(contour)
#         #         #     print(area)
#         #         #     if area <30000 and area >20000:
#         #         #         cntours.append(contour)           
#         #         #     # using drawContours() function
#         #         # print(len(cntours))
#         #     M = cv2.moments(contours_green[1])
#         #     if M['m00'] != 0.0:
#         #         pos_x = int(M['m10']/M['m00'])
#         #         pos_y = int(M['m01']/M['m00'])
#         #     cv2.circle(imageFrame, (pos_x, pos_y),3,(0, 0, 255),-1)
#         #     if pos_x < 399:
#         #         print("I am in pos_x < 350" )
#         #         cmd_publish = Twist()
#         #         # cmd_publish.linear.y= -0.04
#         #         # cmd_publish.linear.x = 0.02
#         #         cmd_publish.angular.z = 0.02
#         #         self.publisher_.publish(cmd_publish)
#         #         time.sleep(0.1)
#         #         # cmd_publish.linear.y= 0.0
#         #         cmd_publish.angular.z = 0.0
#         #         # cmd_publish.linear.x = 0.0
#         #         self.publisher_.publish(cmd_publish) 
#         #     if pos_x> 401:
#         #         print("I am in pos_x > 450" )
#         #         cmd_publish = Twist()
#         #         # cmd_publish.linear.y= +0.04
#         #         # cmd_publish.linear.x = 0.02
#         #         cmd_publish.angular.z = -0.02
#         #         self.publisher_.publish(cmd_publish)
#         #         time.sleep(0.1)
#         #         cmd_publish.linear.y= 0.04
#         #         cmd_publish.linear.x = 0.0
#         #         cmd_publish.angular.z = 0.0
#         #         self.publisher_.publish(cmd_publish) 
#         #     if (pos_x>399 and pos_x<401):
#         #         # global k
#         #         print("I am in the last")
#         #         # k = 0
#         #         cmd_publish = Twist()
#         #         # cmd_publish.angular.z = 0.
#         #         cmd_publish.linear.x = 0.5
                
#         #         self.publisher_.publish(cmd_publish)
#         #         time.sleep(2)
#         #         cmd_publish.linear.x = 0.0
#         #         self.publisher_.publish(cmd_publish)
                
#         # else:
#         #     # global k
#         #     # k+=1
#         #     print("not detected so rotating for detection")
#         #     cmd_publish = Twist()
#         #     cmd_publish.angular.pos_z = 0.15
#         #     self.publisher_.publish(cmd_publish)
#         #     time.sleep(0.1)
#         #     cmd_publish.angular.z = 0.0
#         #     self.publisher_.publish(cmd_publish)
#         cv2.imshow("image", imageFrame)
#         cv2.waitKey(1)                                

        
        
#         # # cntrs = list(contours)
#         # # list for storing names of shapes
#         # cntours = []
#         # for pos,contour in enumerate(contours):
#         #     area = cv2.contourArea(contour)
#         #     print(area)
#         #     if area > 5000:
#         #         cntours.append(contour)           
#         #     # using drawContours() function
#         # print(len(cntours))
       
#         # cv2.drawContours(res_green, [cntours[1]], 0, (0, 0, 255), 5)
        
#         #     # finding center point of shape
#         # M = cv2.moments(contour)
#         # if M['m00'] != 0.0:
#         #     x = int(M['m10']/M['m00'])
#         #     y = int(M['m01']/M['m00'])
#         # cv2.circle(res_green, (x, y),3,(0, 0, 255),-1)
#         # cv2.imshow("image", res_green)
#         # cv2.waitKey(1)

#     def shape_detection(self, image):
#         h, w = image.shape[0:2]
#         neww = 600
#         newh = int(neww*(h/w))
#         image = cv2.resize(image, (neww, newh))
        
#         # converting image into grayscale image
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # setting threshold of gray image
#         _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
#         # using a findContours() function
#         contours, _ = cv2.findContours(
#             threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         i = 0
        
#         # list for storing names of shapes
#         for contour in contours:
        
#             # here we are ignoring first counter because 
#             # findcontour function detects whole image as shape
#             if i == 0:
#                 i = 1
#                 continue
        
#             # cv2.approxPloyDP() function to approximate the shape
#             approx = cv2.approxPolyDP(
#                 contour, 0.01 * cv2.arcLength(contour, True), True)
            
#             # using drawContours() function
#             cv2.drawContours(image, [contour], 0, (0, 0, 255), 5)
        
#             # finding center point of shape
#             M = cv2.moments(contour)
#             if M['m00'] != 0.0:
#                 x = int(M['m10']/M['m00'])
#                 y = int(M['m01']/M['m00'])
        
#             # putting shape name at center of each shape
#             if len(approx) == 4:
#                 cv2.putText(image, 'Quadrilateral', (x, y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                 return 1
#             else:
#                 cv2.putText(image, 'circle', (x, y),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
#                 return 2
            

    
#     def image_sub_callback(self, data):
#         # print("Image received")
#         try:
#             # Convert your ROS Image message to OpenCV2
#             self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
#             # Resize the image
#             # self.image = cv.resize(self.image, (960, 720))
#         except CvBridgeError as e:
#             print(e)
    
    

# def main(args=None):
#     global k 
#     k = 0
#     rclpy.init(args=args)
#     tello = Tello()
#     rclpy.spin(tello) 
#     tello.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()




