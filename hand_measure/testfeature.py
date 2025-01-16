import cv2
import mediapipe as mp
import numpy as np
import glob
import math

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Function to calibrate the camera
def calibrate(dirpath, prefix, image_format, square_size, width=9, height=6):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if dirpath[-1:] == '/':
        dirpath = dirpath[:-1]

    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]

# Chessboard square size real life in cm
square_size = 2.5  # cm
width = 9
height = 6
ret, mtx, dist, rvecs, tvecs = calibrate("static/calibration/", "image", "png", square_size, width, height)

# Load the image
image_path = 'static/uploads/imagetest.jpg'  # Replace with your image file path
imageRead = cv2.imread(image_path)

# Undistort the image 
h, w = imageRead.shape[:2]

# The picture might not have same resolution as the calibration images
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
image = cv2.undistort(imageRead, mtx, dist, None, new_mtx)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_img = image.copy()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Process the image for hand landmarks
results = hands.process(image_rgb)

# Function to apply skin mask and then use CannyEdge detection
def canny_edge_detection(img, blur_ksize=5, threshold1=50, threshold2=290):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinMask= cv2.inRange(hsvim, lower, upper)

    # blur the mask to help remove noise
    skinMask= cv2.blur(skinMask, (2, 2))

    # get threshold image
    _, thresh = cv2.threshold(skinMask, 100, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)

    img_gaussian = cv2.GaussianBlur(thresh, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(img_gaussian, threshold1, threshold2)

canny_img = canny_edge_detection(image.copy())

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to crop a region based on specific landmarks
def crop_hand_region_from_canny(canny_image, landmarks, points):
    h, w = canny_image.shape  # Only two dimensions for Canny (grayscale/binary)

    # Get pixel coordinates of specified landmarks
    coordinates = [(int(landmarks[point].x * w), int(landmarks[point].y * h)) for point in points]

    # Compute the bounding rectangle
    x_coords = [pt[0] for pt in coordinates]
    y_coords = [pt[1] for pt in coordinates]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Crop the region of interest from the Canny image
    cropped_canny = canny_image[y_min:y_max, x_min:x_max]

    return cropped_canny, (x_min, y_min), coordinates

# Function to find the largest contour in a cropped image
def find_largest_contour(cropped_image):
    contours, _ = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None

# Function to find the lowest point of a contour (i.e., the point with the highest y-coordinate)
def find_lowest_point(contour):
    # Initialize variables for the lowest point
    lowest_point = None
    max_y = -1  # Start with a very low y value (since y coordinates increase downward)
    
    # Iterate over the contour points
    for point in contour:
        x, y = point[0][0], point[0][1]  # Get the x, y coordinates
        if y > max_y:  # We are looking for the highest y value (lowest point)
            max_y = y
            lowest_point = (x, y)
    
    return lowest_point

# Function to calculate intersection with Canny edges
def find_intersection_with_canny(canny_image, point1, point2):
    h, w = canny_image.shape  # Height and width of the Canny edge image

    # Convert MediaPipe normalized landmarks to pixel coordinates
    x1, y1 = int(point1.x * w), int(point1.y * h)
    x2, y2 = int(point2.x * w), int(point2.y * h)

    # Calculate the slope (m) and y-intercept (c) of the line
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    else:
        slope = None  # Vertical line

    # Iterate along the line upward to find the intersection with Canny edges
    for y in range(y1, 0, -1):  # Move upward from y1 to the top of the image
        if slope is not None:
            x = int((y - intercept) / slope)
        else:
            x = x1  # Vertical line case

        if 0 <= x < w and canny_image[y, x] != 0:  # Check for edge pixel
            return (x, y)
    return None

def find_perpendicular_intersection(cannyimg, image, mediapipe_pairs, lowest_points, hand_landmarks):
    h, w = image.shape[:2]  # Get image dimensions

    for i, (pair, lowest_point) in enumerate(zip(mediapipe_pairs, lowest_points)):
        # Get the coordinates of the pair of points from MediaPipe landmarks
        p1 = hand_landmarks.landmark[pair[0]]
        p2 = hand_landmarks.landmark[pair[1]]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)

        # Get the coordinates of the current lowest point
        x3, y3 = lowest_point

        # Calculate the slope (m1) and y-intercept (c1) of the line from the MediaPipe pair
        if x2 != x1:
            m1 = (y2 - y1) / (x2 - x1)
            c1 = y1 - m1 * x1
        else:
            m1 = None  # Vertical line

        # Calculate the slope (m2) of the perpendicular line
        if m1 is not None:
            m2 = -1 / m1  # Negative reciprocal of m1
        else:
            m2 = 0  # Perpendicular line to a vertical line is horizontal

        # Calculate the y-intercept (c2) of the perpendicular line passing through (x3, y3)
        if m1 is not None:
            c2 = y3 - m2 * x3
        else:
            c2 = y3  # For horizontal line, y-intercept is the y-coordinate

        # Find the intersection point between the two lines
        if m1 is not None:
            xi = (c2 - c1) / (m1 - m2)
            yi = m1 * xi + c1
        else:  # Special case: the MediaPipe line is vertical
            xi = x1  # x-coordinate of vertical line
            yi = m2 * xi + c2

        # Draw the intersection point on the result image
        intersection_point = (int(xi), int(yi))
        cv2.circle(result_img, intersection_point, 5, (255, 0, 0), -1)  # Blue dot
        print(f"Intersection point for pair {pair} and lowest point {i + 1}: {intersection_point}")

        # Save the first intersection point
        if i == 0:
            first_intersection_point = intersection_point

        # Additional logic for the first intersection point
        if i == 0 and first_intersection_point:
            x3, y3 = first_intersection_point  # Use the first intersection point
            dx = 1 if m2 >= 0 else -1  # Direction of movement for x
            found_intersection = None

            x, y = x3, y3
            # Extend the perpendicular line pixel by pixel
            while 0 <= x < w and 0 <= y < h:
                x += dx
                y = int(m2 * x + c2)

                # Ensure the coordinates are within bounds
                if 0 <= x < w and 0 <= y < h and cannyimg[y, x] != 0:  # Use canny_image here
                    found_intersection = (x, y)
                    break

            # Draw the extended intersection point if found
            if found_intersection:
                cv2.circle(image, found_intersection, 5, (0, 255, 0), -1)  # Green dot for extended intersection
                print(f"Extended intersection with Canny edge for pair {pair}: {found_intersection}")

def find_perpendicular_cut_points(canny_img, result_img, point1, point2):
    h, w = canny_img.shape  # Dimensions of the image

    # Convert MediaPipe normalized landmarks to pixel coordinates
    x1, y1 = int(point1.x * w), int(point1.y * h)
    x2, y2 = int(point2.x * w), int(point2.y * h)

    # Calculate slope (m) and y-intercept (c) for the line between the two points
    if x2 != x1:
        slope = (y2 - y1) / (x2 - x1)
    else:
        slope = None  # Vertical line

    # Calculate the perpendicular slope
    if slope is not None and slope != 0:
        perp_slope = -1 / slope
    elif slope is None:  # Vertical line case
        perp_slope = 0  # Perpendicular to vertical is horizontal
    else:
        perp_slope = float('inf')  # Perpendicular to horizontal is vertical

    # Extend the perpendicular line in two directions
    def extend_line(x_start, y_start, slope, direction, max_length=1000):
        x, y = x_start, y_start
        while 0 <= x < w and 0 <= y < h:
            if slope == float('inf'):  # Vertical line
                y += direction
            elif slope == 0:  # Horizontal line
                x += direction
            else:
                x += direction
                y = int(y_start + slope * (x - x_start))

            if 0 <= x < w and 0 <= y < h and canny_img[y, x] != 0:
                return (x, y)  # Return the first intersection point
        return None

    # Find the two intersection points
    intersection_left = extend_line(x1, y1, perp_slope, direction=-1)  # Extend left
    intersection_right = extend_line(x1, y1, perp_slope, direction=1)  # Extend right

    # Draw the intersection points
    if intersection_left:
        cv2.circle(result_img, intersection_left, 5, (0, 255, 255), -1)  # Yellow circle
        print(f"Left Intersection: {intersection_left}")
    if intersection_right:
        cv2.circle(result_img, intersection_right, 5, (255, 255, 0), -1)  # Cyan circle
        print(f"Right Intersection: {intersection_right}")

    return intersection_left, intersection_right

def find_and_draw_point(canny_img, result_img, hand_landmarks, point_17_idx=17, point_0_idx=0):
    """
    Finds a point on the Canny edge line such that:
    x < x of MediaPipe point 17 and y > y of MediaPipe point 0.
    Draws the point on the result image.

    :param canny_img: Grayscale Canny edge image.
    :param result_img: Original image where the point will be drawn.
    :param hand_landmarks: MediaPipe hand landmarks object.
    :param point_17_idx: Index of MediaPipe point 17 (default 17).
    :param point_0_idx: Index of MediaPipe point 0 (default 0).
    """
    h, w = canny_img.shape  # Dimensions of the Canny edge image

    # Extract pixel coordinates for MediaPipe points 17 and 0
    point_17 = hand_landmarks.landmark[point_17_idx]
    point_0 = hand_landmarks.landmark[point_0_idx]

    x_17, y_17 = int(point_17.x * w), int(point_17.y * h)
    x_0, y_0 = int(point_0.x * w), int(point_0.y * h)

    # Find a point on the Canny edge that satisfies the conditions
    found_point = None
    for y in range(h):  # Iterate over rows
        for x in range(w):  # Iterate over columns
            if canny_img[y, x] != 0:  # Check if the pixel is an edge
                if x > x_17 and y > y_0:  # Check conditions
                    found_point = (x, y)
                    break
        if found_point:
            break

    # Draw the found point on the result image
    if found_point:
        cv2.circle(result_img, found_point, 5, (0, 255, 255), -1)  # Yellow dot
        print(f"Found point: {found_point}")
    else:
        print("No point found that satisfies the conditions.")

    return found_point

# Function to calculate the real-world size of one pixel
def calculate_pixel_size(mtx, square_size, width, height):
    """
    Calculate the physical size of one pixel in real-world units.
    
    Parameters:
    - mtx: Camera matrix obtained from calibration.
    - square_size: The real-world size of one chessboard square (e.g., in cm).
    - width: Number of inner corners horizontally on the chessboard.
    - height: Number of inner corners vertically on the chessboard.
    
    Returns:
    - Pixel size in real-world units (e.g., cm/pixel).
    """
    # Compute the distance in pixels between two adjacent corners in x and y directions
    fx = mtx[0, 0]  # Focal length in x-direction (in pixels)
    fy = mtx[1, 1]  # Focal length in y-direction (in pixels)
    
    # Calculate the real-world size of one pixel
    pixel_size_x = square_size / fx
    pixel_size_y = square_size / fy
    
    return pixel_size_x, pixel_size_y
def calculate_real_distance(p1, p2, pixel_size_x, pixel_size_y):
    """
    Calculate the real-world distance between two points.
    """
    dx = (p2[0] - p1[0]) * pixel_size_x  # Convert x distance to real-world units
    dy = (p2[1] - p1[1]) * pixel_size_y  # Convert y distance to real-world units
    distance = math.sqrt(dx**2 + dy**2)  # Euclidean distance
    return distance


# Calculate pixel size
pixel_size_x, pixel_size_y = calculate_pixel_size(mtx, square_size, width, height)
print(f"Pixel size: {pixel_size_x:.5f} cm/pixel (x-axis), {pixel_size_y:.5f} cm/pixel (y-axis)")

p1 = (100, 150)
p2 = (200, 250)

distance = calculate_real_distance(p1, p2, pixel_size_x, pixel_size_y)
print(f"Real-world distance: {distance:.2f} cm")

# Define the MediaPipe pairs
mediapipe_pairs = [(5, 6), (9, 10), (13, 14), (17, 18), (2, 3)]

# Group all points for cropping
points_to_crop = [
    [5, 6, 9, 10],    # Original region
    [9, 10, 13, 14],  # Region 1
    [13, 14, 17, 18], # Region 2
    [1, 2, 5, 6]      # Region 3
]

finger_lines = [
            (7, 8),  # Index finger
            (11, 12),  # Middle finger
            (15, 16),  # Ring finger
            (19, 20),  # Pinky finger
            (3,4),
        ]
# Define the pairs for each finger
finger_pairs = [(6, 5), (10, 9), (14, 13), (18, 17), (3, 4)]
# Lists to store all intersection and lowest points
intersection_points = []
lowest_points = []

# Process hands and all fingers
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for pair in finger_pairs:
            # Get landmarks for the current pair
            point1 = hand_landmarks.landmark[pair[0]]
            point2 = hand_landmarks.landmark[pair[1]]

            # Find the intersection points for the current pair
            intersections = find_perpendicular_cut_points(canny_img, result_img, point1, point2)
            print(f"Intersections for pair {pair}: {intersections}")

        # Draw the hand landmarks on the image
        mp_drawing.draw_landmarks(
            result_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        ) 

    # Process each set of points for cropping
    for i, points in enumerate(points_to_crop):
        # Crop the hand region from the Canny edge-detected image
        cropped_canny, offset, landmark_coords = crop_hand_region_from_canny(canny_img, hand_landmarks.landmark, points)

        # Find the largest contour in the cropped region
        largest_contour = find_largest_contour(cropped_canny)

        # If a contour was found, calculate the lowest point
        if largest_contour is not None:
            # Find the lowest point (highest y-coordinate)
            lowest_point = find_lowest_point(largest_contour)

            # Draw the largest contour on the cropped image (for visualization)
            contour_img = np.zeros_like(cropped_canny)
            cv2.drawContours(contour_img, [largest_contour], -1, 255, 2)

            # Draw the lowest point as a red circle
            if lowest_point is not None:
                # Calculate the adjusted coordinates for the original image
                adjusted_lowest_point = (lowest_point[0] + offset[0], lowest_point[1] + offset[1])
                cv2.circle(result_img, adjusted_lowest_point, 5, (0, 0, 255), -1)  # Red dot for the lowest point
                print(f"Lowest Point of Contour {i + 1} on Original Image: {adjusted_lowest_point}")
            # Append the lowest point to the list
                lowest_points.append(adjusted_lowest_point)
                if i == 1:
                    lowest_points.append(adjusted_lowest_point)

    find_perpendicular_intersection(canny_img, result_img, mediapipe_pairs, lowest_points, hand_landmarks)

    for start, end in finger_lines:
        # Find intersection point for each finger line
        intersection = find_intersection_with_canny(
            canny_img,
            hand_landmarks.landmark[start],
            hand_landmarks.landmark[end]
        )

        # Draw the intersection point if found
        if intersection:
            cv2.circle(result_img, intersection, 5, (0, 255, 0), -1)  # Green dot
            print(f"Intersection point for line {start}-{end}: {intersection}")

            # Append the intersection point to the list
            intersection_points.append(intersection)
    found_point = find_and_draw_point(canny_img, result_img, hand_landmarks)

# Show results
cv2.imshow("Hand Landmarks", result_img)
cv2.imshow("Canny Edges", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate the real-world distances between points

# Length 1: 