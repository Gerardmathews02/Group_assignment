import cv2
import numpy as np
import statistics

#DO NOT CHANGE

############################## IMAGE PROCESSING ##################################

def resize(image):
    scale_percent = 20
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

# Load the image
image = cv2.imread('PXL_20241012_054651758.MP.jpg')
if image is None:
    raise ValueError("Image not found or cannot be opened.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
abs_sobelx = np.absolute(sobelx)
scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
_, thresholded = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresholded, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# cv2.imshow('Filtered Book Spine centres', resize(eroded))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = []
boundaries = []
green_points = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = h / float(w)
    centre_X = x + w / 2
    centre_Y = y + h / 2
    if aspect_ratio > 3 and h > 500 and w > 15:
        bounding_boxes.append((x, y, w, h))
        boundaries.append(contour)
        for point in contour:
            green_points.append((int(point[0][0]), int(point[0][1])))
            cv2.circle(image, (point[0][0], point[0][1]), 3, (0, 255, 0), -1)

############################## LINE FITTING #######################################
lines = []
for contour in boundaries:
    if len(contour) > 1:  
        [vx, vy, x_, y_] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        vx, vy = vx[0], vy[0]
        x_, y_ = x_[0], y_[0]
        
        if abs(vx) > 0.001:  
            slope = float(vy) / float(vx)  
            intercept = float(y_) - (slope * float(x_))
            lines.append((slope, intercept))

            # Draw black lines for book edges
            left_end_y = int(slope * 0 + intercept)
            right_end_y = int(slope * (image.shape[1] - 1) + intercept)
            cv2.line(image, (0, left_end_y), (image.shape[1] - 1, right_end_y), (0, 0, 0), 2)
        else:
            x_ = int(x_)
            cv2.line(image, (x_, 0), (x_, image.shape[0] - 1), (0, 0, 0), 2)

############################## RED POINTS ########################################
green_points = [tuple(green) for green in green_points]
bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])


centres = []
# Calculate centres of bounding boxes
for i in range(len(bounding_boxes) - 1):
    current_x, current_y, current_w, current_h = bounding_boxes[i]
    next_x, next_y, next_w, next_h = bounding_boxes[i + 1]
    centre_x = (current_x + current_w + next_x) // 2
    centre_y = current_y + current_h // 2
    centres.append((centre_x, centre_y))

# Calculate distances and find statistical outliers
distances = [centres[i + 1][0] - centres[i][0] for i in range(len(centres) - 1)]
mean_distance = np.mean(distances)
std_distance = np.std(distances)
filtered_centres = []

current_distance = abs(centres[0][0] - centres[1][0])
if abs(current_distance - mean_distance) <= std_distance:
    filtered_centres.append(centres[0])
    prev_centre_x = filtered_centres[0][0]

for i in range(1, len(centres) - 1):
    current_distance = centres[i][0] - prev_centre_x
    if abs(current_distance - mean_distance) <= std_distance:
        filtered_centres.append(centres[i])
        prev_centre_x = centres[i][0]

if abs(current_distance - mean_distance) <= std_distance:
    filtered_centres.append(centres[i + 1])

# Draw red dots at filtered centres
centre_ys = [centres[i][1] for i in range(len(centres) - 1)]
mean_height = np.mean(centre_ys)
std_height = np.std(centre_ys)
red_points = []
for centre_x, centre_y in filtered_centres:
    if abs(centre_y - mean_height) <= std_height * 1.2:
        red_point = (centre_x, centre_y)
        red_points.append(red_point)
        cv2.circle(image, (centre_x, centre_y), 10, (0, 0, 255), -1)

############################## YELLOW POINTS ######################################

# Place yellow dots where lines cross horizontal line at y = r_y
for r_x, r_y in red_points:
    closest_left = None
    closest_right = None
    closest_left_x = None
    closest_right_x = None

    for slope, intercept in lines:
        # Calculate intersection x-coordinate for given y (r_y)
        x_at_r_y = (r_y - intercept) / slope

        if x_at_r_y < r_x:  # To the left of the red point
            if closest_left is None or abs(x_at_r_y - r_x) < abs(closest_left_x - r_x):
                closest_left = (x_at_r_y, r_y)
                closest_left_x = x_at_r_y
        elif x_at_r_y > r_x:  # To the right of the red point
            if closest_right is None or abs(x_at_r_y - r_x) < abs(closest_right_x - r_x):
                closest_right = (x_at_r_y, r_y)
                closest_right_x = x_at_r_y

    if closest_left:
        cv2.circle(image, (int(closest_left[0]), int(closest_left[1])), 5, (0, 255, 255), -1)
    if closest_right:
        cv2.circle(image, (int(closest_right[0]), int(closest_right[1])), 5, (0, 255, 255), -1)

cv2.imshow('Filtered Book Spine centres', resize(image))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("filtered_spine_centered_output_with_yellow_dots.jpg", image)

