import cv2 as cv
import numpy as np

# load image
image = cv.imread('15.jpg')
image = cv.imread('16.png')
# # Resize image
image = cv.resize(image, (640, 480))

"CODE FOR THE IMAGE 15"
# # Convert the image to HSV
# image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# # Define the upper and lower boundaries of the green color in the HSV color space
# lower_range_green = (30, 100, 100)
# upper_range_green = (90, 255, 255)
# # Generate the mask
# mask = cv.inRange(image, lower_range_green, upper_range_green)
# # Apply morphological operations to remove noise and fill gaps
# kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# mask = cv.erode(mask, kernel, iterations=1)
# mask = cv.dilate(mask, kernel, iterations=1)
# # Extract the objects from the image
# image = cv.bitwise_and(image, image, mask=mask)

# Convert the image to grayscale
image_grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# Blur the image
image_grey = cv.GaussianBlur(image_grey, (5, 5), 0)
# Apply Otsu's thresholding
_, image_grey = cv.threshold(image_grey, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Calculate the contours with RETR_EXTERNAL to only get the outer contours
contours, hierarchies = cv.findContours(image_grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Get only the parent contours
filtered_contours = []
for i, hierarchy in enumerate(hierarchies[0]):
    print(hierarchy)
    if hierarchy[3] == 0:
        print(hierarchy)
        filtered_contours.append(contours[i])

# for i, cnt in enumerate(contours):
#     # cv.drawContours(image, [cnt], 0, (0, 0, 255), 2)
#     # print(f"Contour {i}: {cnt}")
#     rect = cv.minAreaRect(cnt)
#     box = cv.boxPoints(rect)
#     box = np.int0(box)
#     cv.drawContours(image, [box], 0, (0, 0, 255), 2)
#     # Get the center of the rectangle
#     x, y = rect[0]
#     # Draw the center of the rectangle
#     cv.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

# # Fit an ellipse to the contour points
ellipses = []
for i, cnt in enumerate(filtered_contours):
    # print(f"Contour {i}: {cnt}")
    # Make sure the contour has at least 5 points
    if len(cnt) >= 5:
        ellipses.append(cv.fitEllipse(cnt))

# Draw the ellipse
for i, ellipse in enumerate(ellipses):
    # print(f"Ellipse: {ellipse}")
    # Get the ellipse parameters
    (x, y), (d1, d2), angle = ellipse
    print(f"x: {x} y: {y} d1: {d1} d2: {d2} angle: {angle}")

    # Get the major and minor axis
    major_axis = max(d1, d2)
    minor_axis = min(d1, d2)

    # Calculate the area of the contour
    area = cv.contourArea(contours[i])

    # Calculate the perimeter of the contour
    perimeter = cv.arcLength(contours[i], True)

    # Calculate the roundness
    roundness = 4 * np.pi * area / perimeter ** 2

    # Calculate the aspect ratio
    aspect_ratio = major_axis / minor_axis

    # Draw the ellipse
    cv.ellipse(image, ellipse, (0, 255, 0), 3)

    # Draw the major and minor axis
    cv.line(image, (int(x - major_axis / 2), int(y)), (int(x + major_axis / 2), int(y)), (255, 0, 0), 2)
    cv.line(image, (int(x), int(y - minor_axis / 2)), (int(x), int(y + minor_axis / 2)), (255, 0, 0), 2)

    # Draw the center of the ellipse
    cv.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

# Fit a rectangle to the contour points
# rectangles = []
# for i, cnt in enumerate(contours):
#     # print(f"Contour {i}: {cnt}")
#     # Make sure the contour has at least 5 points
#     # if len(cnt) >= 5:
#     rectangles.append(cv.minAreaRect(cnt))

# # Draw the rectangle
# for i, rectangle in enumerate(rectangles):
#     print(f"Rectangle: {rectangle}")
#     box = cv.boxPoints(rectangle)
#     box = np.int0(box)
#     cv.drawContours(image, [box], 0, (0, 0, 255), 2)

# show image
cv.imshow('image', image)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()