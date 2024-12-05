import cv2
import mediapipe as mp
import numpy as np

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
def canny_edge_detection(img, blur_ksize=5, threshold1=100, threshold2=200):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    return cv2.Canny(img_gaussian, threshold1, threshold2)

canny_img = canny_edge_detection(image.copy())

# Initialize MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to process a specific finger
def process_finger(hand_landmarks, dip_landmark, tip_landmark, color, finger_name):
    h, w, _ = image.shape

    # Convert landmarks to pixel coordinates
    dip_pixel = np.array([int(dip_landmark.x * w), int(dip_landmark.y * h)])  # (x, y)
    tip_pixel = np.array([int(tip_landmark.x * w), int(tip_landmark.y * h)])  # (x, y)

    # Calculate the direction vector for the line
    direction = tip_pixel - dip_pixel  # Vector from DIP to TIP

    # Extend the line segment
    scale = max(w, h)  # Large enough to ensure intersection with edges
    extended_start = dip_pixel - direction * scale
    extended_end = tip_pixel + direction * scale

    # Create a binary image for the extended line
    line_img = np.zeros_like(canny_img)
    cv2.line(line_img, tuple(extended_start.astype(int)), tuple(extended_end.astype(int)), 255, 1)

    # Find intersection points with the Canny edges
    intersections = cv2.bitwise_and(line_img, canny_img)
    points = np.column_stack(np.where(intersections > 0))  # Get (y, x) coordinates

    # Find the intersection point closest to the tip direction
    if len(points) > 0:
        tip_point = np.array([tip_pixel[1], tip_pixel[0]])  # (y, x)
        distances = np.linalg.norm(points - tip_point, axis=1)
        closest_point = points[np.argmin(distances)]  # (y, x)
        intersection_point = (closest_point[1], closest_point[0])  # Convert back to (x, y)

        # Mark the intersection on the image
        cv2.circle(result_img, intersection_point, 5, color, -1)
        print(f"{finger_name} finger intersection at: {intersection_point}")
    else:
        print(f"No intersection found for {finger_name} finger.")

# Define finger landmark mappings and colors
finger_info = [
    ("Thumb", mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP, (255, 0, 0)),  # Blue
    ("Index", mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, (0, 255, 0)),  # Green
    ("Middle", mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, (0, 0, 255)),  # Red
    ("Ring", mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP, (255, 255, 0)),  # Cyan
    ("Pinky", mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP, (255, 0, 255))  # Magenta
]

# Process hands and all fingers
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw the hand landmarks on the image
        mp_drawing.draw_landmarks(
            result_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Process all fingers
        for finger_name, dip_landmark_idx, tip_landmark_idx, color in finger_info:
            process_finger(
                hand_landmarks,
                hand_landmarks.landmark[dip_landmark_idx],
                hand_landmarks.landmark[tip_landmark_idx],
                color,
                finger_name
            )

# Show results
cv2.imshow("Hand Landmarks with Intersection Points", result_img)
cv2.imshow("Canny Edges", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
