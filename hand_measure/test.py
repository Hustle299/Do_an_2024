import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Load the image using OpenCV
image_path = 'static/uploads/imagetest.jpg'  # Replace with your image file path
image = cv2.imread(image_path)

result_img = image.copy()

# Convert the BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and get hand landmarks
results = hands.process(image_rgb)

# Draw hand landmarks
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(result_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
def canny_edge_detection(img, blur_ksize=5, threshold1=100, threshold2=200):
    """
    image: image
    blur_ksize: Gaussian kernel size
    threshold1: min threshold 
    threshold2: max threshold
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

    img_canny = cv2.Canny(img_gaussian,threshold1,threshold2)

    return img_canny

sobel_img = canny_edge_detection(image.copy())

# Display the image with landmarks
cv2.imshow('Hand Landmarks', result_img)
cv2.imshow('Hand', image)
cv2.imshow('SOBEL Hand', sobel_img)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()