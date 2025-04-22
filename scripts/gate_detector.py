import cv2 as cv
import numpy as np
from cv_bridge import CvBridgeError

class Mixin:
    def show_image(self, title, image, resize=False, width=640, height=480):
        # Resize the image
        if resize:
            image = cv.resize(image, (width, height))
        cv.imshow(title, image)
        cv.waitKey(1)

    def background_foreground_separator(self, image, lower_range, upper_range):
        # Create a copy of the image
        image = image.copy()
        # image = cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        # Convert the image to HSV
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # Generate the mask
        mask = cv.inRange(image, lower_range, upper_range)
        # Apply morphological operations to remove noise and fill gaps
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)
        # mask = cv.erode(mask, kernel, iterations=2)
        # mask = cv.dilate(mask, kernel, iterations=2)
        # Extract the objects from the image
        image = cv.bitwise_and(image, image, mask=mask)
        # Convert the image to grayscale
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR)
        # Equalize the histogram of the image
        # image = cv.equalizeHist(image)
        # Blur the image to reduce noise
        image = cv.GaussianBlur(image, (5, 5), 0)
        # Apply Adaptive Thresholding
        # image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
        # Apply Otsu's thresholding
        # _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return image

    def edge_detector(self, image):
        if len(image.shape) == 3:
            # Convert the image to grayscale
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Normalize the image
        image = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # Equalize the histogram of the image
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        # Blur the image to reduce noise
        image = cv.GaussianBlur(image, (3, 3), 0)
        # Detecte edges with laplacian of gaussian
        image = cv.Laplacian(image, cv.CV_64F, ksize=3)
        # Convert the image to absolute values
        image = cv.convertScaleAbs(image)
        image = cv.addWeighted(image, 1.5, image, 0, 0)
        # Apply median blur to reduce noise
        image = cv.medianBlur(image, 3)
        # Apply Otsu's thresholding
        image = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, -7)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        image = cv.morphologyEx(image , cv.MORPH_CLOSE, kernel, iterations=1)
        return image

    def gate_detector(self, image):
        # Create a copy of the image
        image = image.copy()
        image = self.edge_detector(image)
        # Calculate the contours of the image 
        contours, hierarchies = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # Sort the contours based on the area of the bounding box
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:10]
        # Reconvert the image to display the contours with color
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # Find the gate
        self.gates = []
        for contour in contours:
            cv.drawContours(image, [contour], 0, (0, 255, 0), 2)
            # rect = cv.minAreaRect(contour)
            # box = cv.boxPoints(rect)
            # box = np.int0(box)
            # cv.drawContours(image,[box],0,(0,0,255),2)
            # Create a bounding box around the contour
            x, y, w, h = cv.boundingRect(contour)
            # Calculate the area of the gate
            area = cv.contourArea(contour)
            area = area / (image.shape[0] * image.shape[1])
            # Calculate the ratio of the bounding box to see if it is a square
            ratio = w / h
            # Draw the bounding box
            # If the ratio is a square and the area is between 2% and 60% of the image
            if 0.75 < ratio < 1.4 and 0.010 < area < 0.55:
                # Calculate the center of the gate
                cx = x + w / 2
                cy = y + h / 2
                # Save the gate
                self.gates.append((x, y, w, h, int(cx), int(cy), np.round(area, 2)))

        # Sort the gates based on the area of the bounding box
        self.gates = sorted(self.gates, key=lambda x: x[6], reverse=True)

        # Update the current and previous gates only if the drone is not moving
        # if not self.moving:
        if len(self.gates) > 0:
            self.curr_gate = self.gates[0]
            self.gate_found = True
            self.searching = False
        else:
            self.curr_gate = None
            self.gate_found = Falsegate_detector.Mixin
            self.searching = True

        if self.curr_gate is not None:
            x, y, w, h, cx, cy, area = self.curr_gate
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            cv.putText(image, "Curr", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(image, "Area: {:.2f}".format(area), (x, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image

    def generate_grid(self, image):
        # Create a copy of the image
        image = image.copy()
        # If the image is not in BGR, convert it
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        # Divide the image into 3 rows and 3 columns
        rows, cols, _ = image.shape
        row_step = rows // 3
        col_step = cols // 3

        # Draw the grid
        for i in range(1, 3):
            cv.line(image, (0, i * row_step), (cols, i * row_step), (255, 255, 0), 5)
            cv.line(image, (i * col_step, 0), (i * col_step, rows), (255, 255, 0), 5)
        
        # Add a dot to the center of the image
        cv.circle(image, (cols // 2, rows // 2), 10, (0, 0, 255), -1)

        # Draw a line from the center of the image to the center of the gate
        if self.curr_gate is not None:
            # Draw a line from the center of the image to the center of the gate
            first_gate = self.curr_gate
            x, y, w, h, cx, cy, _ = first_gate
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 255, 0), 5)
        if self.curr_stop is not None:
            # Draw a line from the center of the image to the center of the stop sign
            stop_sign = self.curr_stop
            x, y, w, h, cx, cy, _ = stop_sign
            cv.line(image, (cx, cy), (cols // 2, rows // 2), (0, 0, 255), 5)
        return image

    def stop_sign_detector(self, image):
        # print("Checking stop sign")
        image = image.copy()
        # Detect edges
        image = self.edge_detector(image)
        # Detect the stop signs
        # detection = self.detector.detectMultiScale(image, scaleFactor = 1.25, minNeighbors = 7, minSize = (80, 80), maxSize = (500, 500))
        # Calculate the contours of the image 
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # Sort the contours based on the area of the bounding box
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)[:5]
        # Reconvert the image to display the contours with color
        if len(image.shape) != 3:
            image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
        # Find the stop signs
        self.stop_signs = []
        # Draw the contours based on the hierarchy of the contours
        for contour in contours:
            # Approximate the contour with a polygon
            epsilon = 0.08*cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,epsilon,True)
            # Check if the approximated shape has 8 sides
            if len(approx) == 4 and cv.isContourConvex(approx):
                # Calculate the area of the contour
                area = cv.contourArea(contour)
                x, y, w, h = cv.boundingRect(contour)
                # Calculate the area of the bounding box
                area = w * h
                area = area / (image.shape[0] * image.shape[1])
                # Calculate the ratio of the bounding box to see if it is a square
                ratio = w / h
                # If the ratio is a square and the area is between 2% and 60% of the image
                # print("STOP SIGN Ratio: {:.2f}, Area: {:.2f}".format(ratio, area))
                if 0.7 < ratio < 1.1 and 0.010 < area < 0.4:
                    # Calculate the center of the bounding box
                    cx = x + w // 2
                    cy = y + h // 2
                    # Save the gate
                    self.stop_signs.append((x, y, w, h, cx, cy, area))

        # Sort the stop signs based on the area of the bounding box
        self.stop_signs = sorted(self.stop_signs, key=lambda x: x[6], reverse=True)

        # Update the current and previous stop signs only if the drone is not moving
        if len(self.stop_signs) > 0: # and not self.moving:
            self.curr_stop = self.stop_signs[0]
            self.stop_sign_found = True
            self.searching = False
        else:
            self.curr_stop = None
            self.stop_sign_found = False
            self.searching = True

        if self.curr_stop is not None:
            x, y, w, h, cx, cy, area = self.curr_stop
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.circle(image, (cx, cy), 5, (0, 0, 255), -1)
            cv.putText(image, "Curr", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv.putText(image, "Area: {:.2f}".format(area), (x, y + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image

    def image_sub_callback(self, data):
        # print("Image received")
        try:
            # Convert your ROS Image message to OpenCV2
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # Resize the image
            # self.image = cv.resize(self.image, (960, 720))
        except CvBridgeError as e:
            print(e)
