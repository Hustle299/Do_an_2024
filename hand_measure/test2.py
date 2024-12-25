import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load the image
image_path = 'static/uploads/imagetest.jpg'  # Replace with your image file path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result_img = image.copy()

# Process the image for hand landmarks
results = hands.process(image_rgb)

# Apply Canny edge detection
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

# Lists to store all intersection and lowest points
intersection_points = []
lowest_points = []

# Process hands and all fingers
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
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

    
# Show results
cv2.imshow("Hand Landmarks", result_img)
cv2.imshow("Canny Edges", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
