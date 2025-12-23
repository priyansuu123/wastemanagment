# import cv2
# import numpy as np
# import csv
# import time
# from pyzbar.pyzbar import decode
# import os

# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time", "Waste"])

# def parse_qr_text(text):
#     text = text.strip()
#     if ";" in text:
#         parts = {}
#         for p in text.split(";"):
#             if ":" in p:
#                 k, v = p.split(":",1)
#                 parts[k.strip().lower()] = v.strip()
#         name = parts.get("name", "Unknown")
#         erp = parts.get("erp", "Unknown")
#         return name, erp
#     return text, "Unknown"

# def classify_waste(plate_area, food_area):
#     if plate_area <= 0:
#         return "NO PLATE"
#     ratio = food_area / plate_area
#     if ratio > 0.6:
#         return "HIGH"
#     elif ratio > 0.25:
#         return "MEDIUM"
#     elif ratio > 0.08:
#         return "LOW"
#     else:
#         return "EMPTY"

# def detect_plate_food(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9,9),1.5)
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
#                                param1=100, param2=30, minRadius=60, maxRadius=400)
#     if circles is None:
#         return None, 0, 0
#     circles = np.uint16(np.around(circles))
#     x, y, r = circles[0][0]
#     x1, y1, x2, y2 = max(0,x-r), max(0,y-r), min(frame.shape[1],x+r), min(frame.shape[0],y+r)
#     plate_roi = frame[y1:y2, x1:x2]
#     if plate_roi.size == 0:
#         return None,0,0
#     hsv = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)
#     lower1, upper1 = np.array([5,50,40]), np.array([35,255,255])
#     lower2, upper2 = np.array([10,30,20]), np.array([25,200,255])
#     mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
#     food_area = cv2.countNonZero(mask)
#     plate_area = plate_roi.shape[0]*plate_roi.shape[1]
#     cv2.circle(frame, (x,y), r, (0,255,0),2)
#     return (x1,y1,x2,y2), plate_area, food_area

# def main():
#     cap = cv2.VideoCapture(0)
#     processed_qrs = set()
#     print("Show student QR to camera. Press 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         qrs = decode(frame)
#         data = None
#         name, erp = None, None
#         if qrs:
#             data = qrs[0].data.decode("utf-8")
#             name, erp = parse_qr_text(data)
#         if data and data not in processed_qrs:
#             processed_qrs.add(data)
#             print(f"QR detected: {name} | {erp}")
#             plate_box, plate_area, food_area = detect_plate_food(frame)
#             if plate_box is None:
#                 status = "NO PLATE"
#                 print("Plate not detected.")
#             else:
#                 status = classify_waste(plate_area, food_area)
#                 print(f"Waste level: {status}")
#             ts = time.localtime()
#             date = time.strftime("%Y-%m-%d", ts)
#             tstr = time.strftime("%H:%M:%S", ts)
#             with open(CSV_FILE,"a",newline="",encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([name,erp,date,tstr,status])
#         cv2.putText(frame,"Show QR | Press q to quit",(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
#         cv2.imshow("Attendance + Food Demo", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#     print("Program ended.")

# if __name__ == "__main__":
#     main()

# import cv2
# import numpy as np
# import csv
# import time
# from pyzbar.pyzbar import decode
# import os

# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time", "Waste"])

# def parse_qr_text(text):
#     text = text.strip()
#     if ";" in text:
#         parts = {}
#         for p in text.split(";"):
#             if ":" in p:
#                 k, v = p.split(":",1)
#                 parts[k.strip().lower()] = v.strip()
#         name = parts.get("name", "Unknown")
#         erp = parts.get("erp", "Unknown")
#         return name, erp
#     return text, "Unknown"

# def classify_waste(plate_area, food_area):
#     if plate_area <= 0:
#         return "NO PLATE"
#     ratio = food_area / plate_area
#     if ratio > 0.6:
#         return "HIGH"
#     elif ratio > 0.25:
#         return "MEDIUM"
#     elif ratio > 0.08:
#         return "LOW"
#     else:
#         return "EMPTY"

# def detect_plate_food(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (9,9),1.5)
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=120,
#                                param1=100, param2=30, minRadius=60, maxRadius=400)
#     if circles is None:
#         return None, 0, 0
#     circles = np.uint16(np.around(circles))
#     # choose largest circle
#     x, y, r = max(circles[0], key=lambda c: c[2])
#     plate_area_estimate = np.pi * r * r
#     if plate_area_estimate < 5000:
#         return None, 0, 0
#     x1, y1, x2, y2 = max(0,x-r), max(0,y-r), min(frame.shape[1],x+r), min(frame.shape[0],y+r)
#     plate_roi = frame[y1:y2, x1:x2]
#     if plate_roi.size == 0:
#         return None, 0, 0
#     hsv = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)
#     lower1, upper1 = np.array([5,50,40]), np.array([35,255,255])
#     lower2, upper2 = np.array([10,30,20]), np.array([25,200,255])
#     mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
#     food_area = cv2.countNonZero(mask)
#     plate_area = plate_roi.shape[0]*plate_roi.shape[1]
#     # plate must cover at least 20% of frame to be valid
#     if plate_area < 0.2 * frame.shape[0]*frame.shape[1]:
#         return None, 0, 0
#     cv2.circle(frame, (x,y), r, (0,255,0),2)
#     return (x1,y1,x2,y2), plate_area, food_area

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera. Try different camera index or grant permission.")
#         return

#     processed_qrs = set()
#     print("Show student QR to camera. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Frame capture failed. Exiting.")
#             break

#         # QR code detection
#         qrs = decode(frame)
#         data = None
#         name, erp = None, None
#         if qrs:
#             data = qrs[0].data.decode("utf-8")
#             name, erp = parse_qr_text(data)

#         if data and data not in processed_qrs:
#             processed_qrs.add(data)
#             print(f"QR detected: {name} | {erp}")
#             print("Please place your plate in front of the camera...")
#             time.sleep(3)  # <-- time for student to place plate

#             # capture a fresh frame for plate detection
#             ret2, frame_plate = cap.read()
#             if not ret2:
#                 print("Failed to capture frame for plate detection.")
#                 continue

#             plate_box, plate_area, food_area = detect_plate_food(frame_plate)
#             if plate_box is None:
#                 status = "NO PLATE"
#                 print("Plate not detected!")
#             else:
#                 status = classify_waste(plate_area, food_area)
#                 print(f"Waste level: {status}")

#             # save attendance
#             ts = time.localtime()
#             date = time.strftime("%Y-%m-%d", ts)
#             tstr = time.strftime("%H:%M:%S", ts)
#             with open(CSV_FILE,"a",newline="",encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([name,erp,date,tstr,status])

#         cv2.putText(frame,"Show QR | Press q to quit",(10,frame.shape[0]-10),
#                     cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
#         cv2.imshow("Attendance + Food Demo", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Quit pressed.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("Program ended.")

# if __name__ == "__main__":
#     main()

# import cv2
# import numpy as np
# import csv
# import time
# from pyzbar.pyzbar import decode
# import os

# # -------------------- CSV SETUP --------------------
# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time", "Waste"])

# # -------------------- QR PARSER --------------------
# def parse_qr_text(text):
#     text = text.strip()
#     if ";" in text:
#         parts = {}
#         for p in text.split(";"):
#             if ":" in p:
#                 k, v = p.split(":", 1)
#                 parts[k.strip().lower()] = v.strip()
#         name = parts.get("name", "Unknown")
#         erp = parts.get("erp", "Unknown")
#         return name, erp
#     return text, "Unknown"

# # -------------------- WASTE CLASSIFIER --------------------
# def classify_waste(plate_area, food_area):
#     if plate_area <= 0:
#         return "NO PLATE"

#     ratio = food_area / plate_area
#     print(f"Food ratio: {ratio:.2f}")

#     if ratio > 0.6:
#         return "HIGH"
#     elif ratio > 0.25:
#         return "MEDIUM"
#     elif ratio > 0.08:
#         return "LOW"
#     else:
#         return "EMPTY"

# # -------------------- STRICT PLATE DETECTION --------------------
# def detect_plate_food(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (11, 11), 2)

#     circles = cv2.HoughCircles(
#         blur,
#         cv2.HOUGH_GRADIENT,
#         dp=1.1,
#         minDist=180,
#         param1=120,
#         param2=25,
#         minRadius=110,
#         maxRadius=260
#     )

#     if circles is None:
#         return None, 0, 0

#     circles = np.uint16(np.around(circles))
#     best_circle = None
#     best_score = 0

#     for (x, y, r) in circles[0]:

#         # ---- SIZE FILTER ----
#         if r < 110 or r > 240:
#             continue

#         # ---- PLATE MASK ----
#         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#         cv2.circle(mask, (x, y), r, 255, -1)

#         plate_pixels = cv2.bitwise_and(frame, frame, mask=mask)

#         # ---- COLOR FILTER (PLATES ARE LIGHT) ----
#         mean_color = cv2.mean(plate_pixels, mask=mask)
#         brightness = np.mean(mean_color[:3])

#         if brightness < 160:
#             continue

#         # ---- EDGE DENSITY FILTER ----
#         edges = cv2.Canny(gray, 80, 160)
#         edge_count = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
#         edge_ratio = edge_count / (np.pi * r * r)

#         if edge_ratio > 0.06:
#             continue

#         score = brightness - (edge_ratio * 1000)
#         if score > best_score:
#             best_score = score
#             best_circle = (x, y, r)

#     if best_circle is None:
#         return None, 0, 0

#     x, y, r = best_circle

#     # ---- FINAL PLATE MASK ----
#     plate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.circle(plate_mask, (x, y), r, 255, -1)

#     plate_area = cv2.countNonZero(plate_mask)
#     plate_only = cv2.bitwise_and(frame, frame, mask=plate_mask)

#     # ---- FOOD DETECTION ----
#     hsv = cv2.cvtColor(plate_only, cv2.COLOR_BGR2HSV)
#     lower_food = np.array([5, 40, 40])
#     upper_food = np.array([35, 255, 255])

#     food_mask = cv2.inRange(hsv, lower_food, upper_food)
#     food_mask = cv2.bitwise_and(food_mask, food_mask, mask=plate_mask)
#     food_area = cv2.countNonZero(food_mask)

#     # ---- DRAW ----
#     food_pct = int((food_area / plate_area) * 100)
#     cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
#     cv2.putText(
#         frame,
#         f"Plate detected | Food {food_pct}%",
#         (x - r, y - r - 10),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (0, 255, 0),
#         2
#     )

#     return (x - r, y - r, x + r, y + r), plate_area, food_area

# # -------------------- MAIN LOOP --------------------
# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open camera.")
#         return

#     processed_qrs = set()
#     print("Show student QR to camera. Press 'q' to quit.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         qrs = decode(frame)
#         data = None
#         name, erp = None, None

#         if qrs:
#             data = qrs[0].data.decode("utf-8")
#             name, erp = parse_qr_text(data)

#         if data and data not in processed_qrs:
#             processed_qrs.add(data)
#             print(f"\nQR detected: {name} | {erp}")
#             print("Place plate in front of camera...")
#             time.sleep(3)

#             ret2, frame_plate = cap.read()
#             if not ret2:
#                 continue

#             plate_box, plate_area, food_area = detect_plate_food(frame_plate)

#             if plate_box is None:
#                 status = "NO PLATE"
#                 print("Plate not detected!")
#             else:
#                 status = classify_waste(plate_area, food_area)
#                 print(f"Waste level: {status}")

#             ts = time.localtime()
#             date = time.strftime("%Y-%m-%d", ts)
#             tstr = time.strftime("%H:%M:%S", ts)

#             with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([name, erp, date, tstr, status])

#         cv2.putText(
#             frame,
#             "Show QR | Press q to quit",
#             (10, frame.shape[0] - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (200, 200, 200),
#             1
#         )

#         cv2.imshow("Attendance + Plate Waste Detection", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("Program ended.")

# # -------------------- RUN --------------------
# if __name__ == "__main__":
#     main()


# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # ---------------- LOAD YOLO SEGMENTATION MODEL ----------------
# model = YOLO("yolov8n-seg.pt")  # lightweight + accurate

# # ---------------- FOOD CLASS IDS (COCO DATASET) ----------------
# FOOD_CLASSES = {
#     46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
#     56, 57, 58, 59  # banana, apple, sandwich, pizza, cake, etc.
# }

# PLATE_CLASS_ID = 60

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Press 'q' to quit")

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, conf=0.4, iou=0.5)

#     plate_mask = None
#     food_mask = None

#     for r in results:
#         if r.masks is None:
#             continue

#         for cls, mask in zip(r.boxes.cls, r.masks.data):
#             cls_id = int(cls)
#             mask_np = mask.cpu().numpy()

#             # -------- PLATE MASK --------
#             if cls_id == PLATE_CLASS_ID:
#                 if plate_mask is None:
#                     plate_mask = mask_np
#                 else:
#                     plate_mask = np.logical_or(plate_mask, mask_np)

#             # -------- FOOD MASK --------
#             if cls_id in FOOD_CLASSES:
#                 if food_mask is None:



    


import cv2
import time

from config import QR_WAIT
from firebase_service import init_firebase, init_csv, save_data
from plate_detection import detect_plate, process_plate
from utils import scan_qr

print("==========================================")
print(" SMART FOOD WASTE DETECTION SYSTEM STARTED ")
print("==========================================")

# ---------- INITIALIZE SERVICES ----------
db = init_firebase()
init_csv()

# ---------- CAMERA SETUP ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("📷 Camera started")
print("👉 Show QR code to camera")

processed_qrs = set()

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not received")
        break

    # ---- QR SCANNING ----
    qr_data, frame = scan_qr(frame)
    cv2.imshow("Food Waste Detection", frame)

    # ---- PROCESS EACH QR ONLY ONCE ----
    if qr_data and qr_data not in processed_qrs:
        processed_qrs.add(qr_data)

        print("🟢 QR detected:", qr_data)
        print("⏳ Waiting for plate placement...")
        time.sleep(QR_WAIT)

        # Capture frame for plate detection
        ret2, plate_frame = cap.read()
        if not ret2:
            print("❌ Could not read plate frame")
            continue

        # ---- PLATE DETECTION (GREEN CONTOUR METHOD) ----
        print("🔍 Detecting plate (green contour method)...")

        center, radius, plate_mask = detect_plate(plate_frame)

        if center is None:
            print("⚠️ Plate not detected. Please place plate properly.")
            continue

        print("✅ Plate detected successfully")

        # ---- FOOD PROCESSING ----
        overlay, food_percent = process_plate(
            plate_frame,
            center,
            radius,
            plate_mask
        )

        cv2.imshow("Food Waste Detection", overlay)
        cv2.waitKey(1500)

        # ---- SAVE DATA ----
        try:
            student_info = qr_data.split(",")
            name = student_info[0]
            erp = student_info[1] if len(student_info) > 1 else "Unknown"

            print("📊 Food waste calculated:", food_percent, "%")

            save_data(db, name, erp, food_percent)

        except Exception as e:
            print("❌ QR format error. Use: Name,ERP")

    # ---- EXIT KEY ----
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Program stopped by user")
        break

# ---------- CLEANUP ----------
cap.release()
cv2.destroyAllWindows()
print("✅ Camera released and windows closed")
