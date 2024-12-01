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

# Check for hands and process middle finger landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        # Get middle finger's last two landmarks
        dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Convert landmarks to pixel coordinates
        h, w, _ = image.shape
        dip_pixel = np.array([int(dip.x * w), int(dip.y * h)])  # (x, y)
        tip_pixel = np.array([int(tip.x * w), int(tip.y * h)])  # (x, y)
        
        # Calculate the direction vector for the line
        direction = tip_pixel - dip_pixel  # Vector from DIP to TIP
        direction = direction / np.linalg.norm(direction)  # Normalize vector

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
            cv2.circle(result_img, intersection_point, 5, (0, 255, 0), -1)
            
            print(f"Intersection at: {intersection_point}")

# Show results
cv2.imshow("Intersection Point", result_img)
cv2.imshow("Canny Edges", canny_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
