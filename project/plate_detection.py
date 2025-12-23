import cv2
import numpy as np

def detect_plate(frame):
    """
    Optimized Hough Circle plate detection
    - Frame resized for speed
    - Only called after QR scan
    - Returns center, radius, and mask
    """

    # Resize to speed up
    h, w = frame.shape[:2]
    scale = 0.5
    small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=int(50*scale),
        maxRadius=int(150*scale)
    )

    if circles is None:
        return None, None, None

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0, 0]

    # Scale back to original frame size
    center = (int(x/scale), int(y/scale))
    radius = int(r/scale)

    # Plate mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    return center, radius, mask


def process_plate(frame, center, radius, plate_mask):
    overlay = frame.copy()

    # Food detection inside mask
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_food = np.array([0, 50, 50])
    upper_food = np.array([179, 255, 200])
    food_mask = cv2.inRange(hsv, lower_food, upper_food)

    food_inside_plate = cv2.bitwise_and(food_mask, plate_mask)
    food_inside_plate = cv2.morphologyEx(
        food_inside_plate,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    # Draw green circle
    cv2.circle(overlay, center, radius, (0, 255, 0), 3)

    plate_area = cv2.countNonZero(plate_mask)
    food_area = cv2.countNonZero(food_inside_plate)
    food_percent = int((food_area / plate_area) * 100) if plate_area > 0 else 0

    cv2.putText(
        overlay,
        f"Food Left: {food_percent}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return overlay, food_percent
