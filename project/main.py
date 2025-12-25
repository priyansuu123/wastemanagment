# import cv2
# import numpy as np
# import time
# from pyzbar.pyzbar import decode
# import csv
# import os
# import threading
# from firebase_service import init_firebase, save_data

# print("🚀 SYSTEM RUNNING (STABLE + FOOD ONLY + HUMAN IGNORE)")

# # ================= CAMERA =================
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# # ================= STATE =================
# state = "SCAN_QR"
# student_id = None
# qr_time = 0
# saved = False

# CSV_FILE = "waste_log.csv"

# # ================= CSV =================
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="") as f:
#         csv.writer(f).writerow(["ERP", "Waste", "Time"])

# # ================= FIREBASE =================
# db = init_firebase()

# # ================= QR =================
# def scan_qr(frame):
#     for obj in decode(frame):
#         data = obj.data.decode("utf-8")
#         x, y, w, h = obj.rect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, data, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#         return data
#     return None

# # ================= PLATE DETECTION (HUMAN IGNORE) =================
# def detect_plate(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (11, 11), 2)

#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1.1,
#         minDist=180,
#         param1=50,
#         param2=25,
#         minRadius=110,
#         maxRadius=240
#     )

#     if circles is None:
#         return None

#     # Only take circles roughly in center (ignore humans/edges)
#     height, width = frame.shape[:2]
#     for c in np.uint16(np.around(circles[0])):
#         x, y, r = c
#         if (width*0.2 < x < width*0.8) and (height*0.2 < y < height*0.8):
#             return x, y, r

#     return None

# # ================= FOOD DETECTION (FOOD ONLY) =================
# def food_detection(frame, plate):
#     x, y, r = plate

#     mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.circle(mask, (x, y), r, 255, -1)

#     plate_only = cv2.bitwise_and(frame, frame, mask=mask)

#     hsv = cv2.cvtColor(plate_only, cv2.COLOR_BGR2HSV)
#     h, s, v = cv2.split(hsv)

#     # Filter only realistic food colors
#     food_mask = cv2.inRange(hsv,
#                             (0, 40, 40),
#                             (179, 255, 255))
#     food_mask = cv2.bitwise_and(food_mask, mask)

#     # Remove small noise blobs
#     kernel = np.ones((3,3), np.uint8)
#     food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_DILATE, kernel, iterations=1)

#     food_pixels = cv2.countNonZero(food_mask)
#     plate_pixels = cv2.countNonZero(mask)

#     # Small pixels → 0%
#     if food_pixels < 600:
#         waste = 0
#     else:
#         ratio = food_pixels / plate_pixels
#         if ratio < 0.08:
#             waste = 15
#         elif ratio < 0.18:
#             waste = 35
#         else:
#             waste = 60

#     # Blue overlay for food
#     overlay = frame.copy()
#     overlay[food_mask > 0] = (255, 0, 0)
#     frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

#     return frame, waste

# # ================= SAVE =================
# def save_result(erp, waste):
#     print("\n==============================")
#     print(f"👤 ERP     : {erp}")
#     print(f"🍽️ WASTE  : {waste}%")
#     print("==============================\n")

#     with open(CSV_FILE, "a", newline="") as f:
#         csv.writer(f).writerow([erp, f"{waste}%", time.strftime("%H:%M:%S")])

#     save_data(db, erp, waste)

# # ================= MAIN LOOP =================
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     if state == "SCAN_QR":
#         qr = scan_qr(frame)
#         cv2.putText(frame, "SCAN STUDENT QR",
#                     (160, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     1,
#                     (0, 255, 255),
#                     2)

#         if qr:
#             student_id = qr
#             qr_time = time.time()
#             saved = False
#             state = "DETECT"

#     elif state == "DETECT":
#         # 3.5 sec wait after QR
#         if time.time() - qr_time < 3.5:
#             cv2.putText(frame, "WAIT... PLACE PLATE",
#                         (140, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1,
#                         (0, 0, 255),
#                         2)
#             cv2.imshow("Food Waste Detection", frame)
#             cv2.waitKey(1)
#             continue

#         plate = detect_plate(frame)

#         if plate is None:
#             # No plate → 0% waste and save
#             cv2.putText(frame, "WASTE: 0%",
#                         (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1.2,
#                         (0, 255, 255),
#                         3)
#             if not saved:
#                 threading.Thread(
#                     target=save_result,
#                     args=(student_id, 0),
#                     daemon=True
#                 ).start()
#                 saved = True
#                 state = "SCAN_QR"
#         else:
#             x, y, r = plate
#             cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
#             frame, waste = food_detection(frame, plate)

#             cv2.putText(frame, f"WASTE: {waste}%",
#                         (20, 80),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1.2,
#                         (0, 255, 255),
#                         3)

#             if not saved:
#                 threading.Thread(
#                     target=save_result,
#                     args=(student_id, waste),
#                     daemon=True
#                 ).start()
#                 saved = True
#                 state = "SCAN_QR"

#     cv2.imshow("Food Waste Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# if __name__ == "__main__":
#     db = init_firebase()
#     save_data(db, "TEST_ERP", 25)






import cv2
import numpy as np
import time
from pyzbar.pyzbar import decode
import csv
import os
import threading
from firebase_service import init_firebase, save_data

print("🚀 SYSTEM RUNNING (STABLE + FOOD ONLY + STEEL PLATE FIXED)")

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ================= STATE =================
state = "SCAN_QR"
student_id = None
qr_time = 0
saved = False

CSV_FILE = "waste_log.csv"

# ================= CSV =================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["ERP", "Waste", "Time"])

# ================= FIREBASE =================
db = init_firebase()

# ================= QR SCAN =================
def scan_qr(frame):
    for obj in decode(frame):
        data = obj.data.decode("utf-8")
        x, y, w, h = obj.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, data, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return data
    return None

# ================= PLATE DETECTION =================
def detect_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=180,
        param1=50,
        param2=25,
        minRadius=110,
        maxRadius=240
    )

    if circles is None:
        return None

    h, w = frame.shape[:2]
    for c in np.uint16(np.around(circles[0])):
        x, y, r = c
        if w*0.25 < x < w*0.75 and h*0.25 < y < h*0.75:
            return x, y, r

    return None

# ================= FOOD DETECTION (FIXED) =================
def food_detection(frame, plate):
    x, y, r = plate

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    plate_only = cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(plate_only, cv2.COLOR_BGR2HSV)

    # FOOD ONLY (ignore steel reflections)
    food_mask = cv2.inRange(
        hsv,
        (0, 60, 50),     # saturation threshold
        (179, 255, 230)  # brightness cap
    )
    food_mask = cv2.bitwise_and(food_mask, mask)

    kernel = np.ones((3,3), np.uint8)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Remove steel shine
    v_channel = hsv[:, :, 2]
    food_mask[v_channel > 230] = 0

    food_pixels = cv2.countNonZero(food_mask)
    plate_pixels = cv2.countNonZero(mask)

    ratio = food_pixels / plate_pixels if plate_pixels > 0 else 0

    # REALISTIC WASTE %
    if ratio < 0.05:
        waste = 0
    elif ratio < 0.12:
        waste = 20
    elif ratio < 0.25:
        waste = 40
    else:
        waste = 60

    print(f"[DEBUG] food_px={food_pixels} plate_px={plate_pixels} ratio={ratio:.3f}")

    overlay = frame.copy()
    overlay[food_mask > 0] = (255, 0, 0)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    return frame, waste

# ================= SAVE =================
def save_result(erp, waste):
    print("\n==============================")
    print(f"👤 ERP     : {erp}")
    print(f"🍽️ WASTE  : {waste}%")
    print("==============================\n")

    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([erp, f"{waste}%", time.strftime("%H:%M:%S")])

    try:
        save_data(db, erp, waste)
    except Exception as e:
        print("❌ Firebase save error:", e)

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if state == "SCAN_QR":
        qr = scan_qr(frame)
        cv2.putText(frame, "SCAN STUDENT QR", (150, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if qr:
            student_id = qr
            qr_time = time.time()
            saved = False
            state = "DETECT"

    elif state == "DETECT":
        if time.time() - qr_time < 3:
            cv2.putText(frame, "WAIT... PLACE PLATE", (120, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Food Waste Detection", frame)
            cv2.waitKey(1)
            continue

        plate = detect_plate(frame)

        if plate:
            x, y, r = plate
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            frame, waste = food_detection(frame, plate)
        else:
            waste = 0

        cv2.putText(frame, f"WASTE: {waste}%", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)

        if not saved:
            threading.Thread(
                target=save_result,
                args=(student_id, waste),
                daemon=True
            ).start()
            saved = True
            state = "SCAN_QR"

    cv2.imshow("Food Waste Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
