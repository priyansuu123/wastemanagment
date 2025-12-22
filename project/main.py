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
#                     food_mask = mask_np
#                 else:
#                     food_mask = np.logical_or(food_mask, mask_np)

#     if plate_mask is not None:
#         plate_area = np.sum(plate_mask)

#         if food_mask is not None:
#             # Food only inside plate
#             food_on_plate = np.logical_and(food_mask, plate_mask)
#             food_area = np.sum(food_on_plate)
#         else:
#             food_area = 0

#         waste_pct = (food_area / plate_area) * 100

#         # -------- VISUALIZATION --------
#         plate_overlay = frame.copy()
#         plate_overlay[plate_mask] = (0, 255, 0)
#         frame = cv2.addWeighted(plate_overlay, 0.3, frame, 0.7, 0)

#         food_overlay = frame.copy()
#         if food_area > 0:
#             food_overlay[food_on_plate] = (0, 0, 255)
#             frame = cv2.addWeighted(food_overlay, 0.4, frame, 0.6, 0)

#         cv2.putText(
#             frame,
#             f"Food Waste: {waste_pct:.2f}%",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#     else:
#         cv2.putText(
#             frame,
#             "No plate detected",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )

#     cv2.imshow("Segmentation-Based Food Waste Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import time

# # ---------------- LOAD MODEL ----------------
# model = YOLO("yolov8n-seg.pt")

# # COCO IDs
# PLATE_CLASS_ID = 60
# FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Press 'q' or ESC to quit")

# last_infer_time = 0
# INFER_DELAY = 0.5   # seconds (limits YOLO speed)

# # ---------------- MAIN LOOP ----------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     current_time = time.time()

#     # --------- THROTTLE YOLO INFERENCE ---------
#     if current_time - last_infer_time < INFER_DELAY:
#         cv2.imshow("Food Waste Segmentation", frame)
#         if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
#             break
#         continue

#     last_infer_time = current_time

#     results = model(frame, conf=0.4, iou=0.5, verbose=False)

#     plate_mask = None
#     food_mask = None

#     for r in results:
#         if r.masks is None:
#             continue

#         for cls, mask in zip(r.boxes.cls, r.masks.data):
#             cls_id = int(cls)
#             mask_np = mask.cpu().numpy().astype(bool)

#             if cls_id == PLATE_CLASS_ID:
#                 plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)

#             if cls_id in FOOD_CLASSES:
#                 food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

#     # --------- CALCULATION ---------
#     if plate_mask is not None:
#         plate_area = np.sum(plate_mask)

#         if plate_area > 0 and food_mask is not None:
#             food_on_plate = np.logical_and(food_mask, plate_mask)
#             food_area = np.sum(food_on_plate)
#             waste_pct = (food_area / plate_area) * 100
#         else:
#             waste_pct = 0

#         # --------- DRAW ---------
#         overlay = frame.copy()
#         overlay[plate_mask] = (0, 255, 0)
#         frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

#         if food_mask is not None:
#             overlay2 = frame.copy()
#             overlay2[food_on_plate] = (0, 0, 255)
#             frame = cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0)

#         cv2.putText(
#             frame,
#             f"Food Waste: {waste_pct:.2f}%",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )
#     else:
#         cv2.putText(
#             frame,
#             "No plate detected",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )

#     cv2.imshow("Food Waste Segmentation", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()
# print("Program ended safely.")

# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO

# # ---------------- YOLO MODEL ----------------
# model = YOLO("yolov8n-seg.pt")

# # COCO class IDs
# PLATE_CLASS_ID = 60
# FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Press Q or ESC to quit")

# # Throttle inference (VERY IMPORTANT)
# INFER_DELAY = 0.6
# last_infer_time = 0

# # ---------------- MAIN LOOP ----------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     now = time.time()
#     if now - last_infer_time < INFER_DELAY:
#         cv2.imshow("Steel Plate Food Waste Detection", frame)
#         if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
#             break
#         continue

#     last_infer_time = now

#     # ---------- YOLO SEGMENTATION ----------
#     results = model(frame, conf=0.15, iou=0.5, verbose=False)

#     plate_mask = None
#     food_mask = None

#     for r in results:
#         if r.masks is None:
#             continue

#         for cls, mask in zip(r.boxes.cls, r.masks.data):
#             cls_id = int(cls)
#             mask_np = mask.cpu().numpy().astype(bool)

#             if cls_id == PLATE_CLASS_ID:
#                 plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)

#             if cls_id in FOOD_CLASSES:
#                 food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

#     # ---------- FALLBACK: STEEL PLATE VIA CIRCLE ----------
#     if plate_mask is None:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (9, 9), 2)

#         circles = cv2.HoughCircles(
#             blur,
#             cv2.HOUGH_GRADIENT,
#             dp=1.1,
#             minDist=200,
#             param1=120,
#             param2=30,
#             minRadius=120,   # tuned for ~30 cm plate
#             maxRadius=240
#         )

#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             x, y, r = max(circles[0], key=lambda c: c[2])

#             # create boolean mask for the plate
#             plate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#             cv2.circle(plate_mask, (x, y), r, 255, -1)
#             plate_mask = plate_mask.astype(bool)

#             # draw circle for visualization
#             cv2.circle(frame, (x, y), r, (0, 255, 0), 3)

#     # ---------- CALCULATE WASTE ----------
#     if plate_mask is not None:
#         plate_area = np.sum(plate_mask)

#         if plate_area > 0 and food_mask is not None:
#             food_on_plate = np.logical_and(food_mask, plate_mask)
#             food_area = np.sum(food_on_plate)
#             waste_pct = (food_area / plate_area) * 100
#         else:
#             waste_pct = 0

#         # ---------- VISUALIZATION ----------
#         overlay = frame.copy()
#         overlay[plate_mask] = (0, 255, 0)
#         frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

#         if food_mask is not None and np.sum(food_on_plate) > 0:
#             overlay2 = frame.copy()
#             overlay2[food_on_plate] = (0, 0, 255)
#             frame = cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0)

#         cv2.putText(
#             frame,
#             f"Food Waste: {waste_pct:.2f}%",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#     else:
#         cv2.putText(
#             frame,
#             "Plate not detected",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )

#     cv2.imshow("Steel Plate Food Waste Detection", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()
# print("Program ended safely.")


# import cv2
# import numpy as np
# import time
# from ultralytics import YOLO

# # ---------------- YOLO MODEL ----------------
# model = YOLO("yolov8n-seg.pt")

# # COCO class IDs
# PLATE_CLASS_ID = 60
# FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Press Q or ESC to quit")

# # Throttle inference
# INFER_DELAY = 0.6
# last_infer_time = 0

# # Minimum time to show the frame after detection (seconds)
# SHOW_TIME = 2.0

# # ---------------- MAIN LOOP ----------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     now = time.time()
#     if now - last_infer_time < INFER_DELAY:
#         cv2.imshow("Steel Plate Food Waste Detection", frame)
#         if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
#             break
#         continue

#     last_infer_time = now

#     # ---------- YOLO SEGMENTATION ----------
#     results = model(frame, conf=0.15, iou=0.5, verbose=False)

#     plate_mask = None
#     food_mask = None

#     for r in results:
#         if r.masks is None:
#             continue
#         for cls, mask in zip(r.boxes.cls, r.masks.data):
#             cls_id = int(cls)
#             mask_np = mask.cpu().numpy().astype(bool)
#             if cls_id == PLATE_CLASS_ID:
#                 plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)
#             if cls_id in FOOD_CLASSES:
#                 food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

#     # ---------- FALLBACK: STEEL PLATE VIA CIRCLE ----------
#     if plate_mask is None:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (9, 9), 2)
#         circles = cv2.HoughCircles(
#             blur,
#             cv2.HOUGH_GRADIENT,
#             dp=1.1,
#             minDist=200,
#             param1=120,
#             param2=30,
#             minRadius=120,
#             maxRadius=240
#         )
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             x, y, r = max(circles[0], key=lambda c: c[2])
#             plate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#             cv2.circle(plate_mask, (x, y), r, 255, -1)
#             plate_mask = plate_mask.astype(bool)
#             cv2.circle(frame, (x, y), r, (0, 255, 0), 3)

#     # ---------- CALCULATE WASTE ----------
#     if plate_mask is not None:
#         plate_area = np.sum(plate_mask)
#         if plate_area > 0 and food_mask is not None:
#             food_on_plate = np.logical_and(food_mask, plate_mask)
#             food_area = np.sum(food_on_plate)
#             waste_pct = (food_area / plate_area) * 100
#         else:
#             waste_pct = 0

#         # ---------- VISUALIZATION ----------
#         overlay = frame.copy()
#         overlay[plate_mask] = (0, 255, 0)
#         frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

#         if food_mask is not None and np.sum(food_on_plate) > 0:
#             overlay2 = frame.copy()
#             overlay2[food_on_plate] = (0, 0, 255)
#             frame = cv2.addWeighted(overlay2, 0.4, frame, 0.6, 0)

#         cv2.putText(
#             frame,
#             f"Food Waste: {waste_pct:.2f}%",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 255, 0),
#             2
#         )

#         # ---------- PRINT TO TERMINAL ----------
#         print(f"Plate detected! Food Waste: {waste_pct:.2f}%")

#         # ---------- SHOW FRAME FOR SOME TIME ----------
#         cv2.imshow("Steel Plate Food Waste Detection", frame)
#         cv2.waitKey(int(SHOW_TIME * 1000))  # freeze for SHOW_TIME seconds

#     else:
#         cv2.putText(
#             frame,
#             "Plate not detected",
#             (30, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )
#         print("Plate not detected")
#         cv2.imshow("Steel Plate Food Waste Detection", frame)
#         cv2.waitKey(int(SHOW_TIME * 1000))

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()
# print("Program ended safely.")


# integrated version

# import cv2
# import numpy as np
# import time
# import csv
# import os
# from pyzbar.pyzbar import decode
# from ultralytics import YOLO

# # ---------------- CSV ----------------
# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time", "Food_Waste_Percentage"])

# def parse_qr_text(text):
#     """Parse QR text formatted as 'Name:John;ERP:12345'"""
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

# # ---------------- YOLO MODEL ----------------
# model = YOLO("yolov8n-seg.pt")

# PLATE_CLASS_ID = 60
# FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Show QR code to camera. Press Q or ESC to quit.")

# # Throttle inference
# INFER_DELAY = 0.6
# last_infer_time = 0

# # Minimum time to show detection frame
# SHOW_TIME = 2.5

# processed_qrs = set()

# # ---------------- MAIN LOOP ----------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # -------- QR DETECTION --------
#     qrs = decode(frame)
#     qr_data = None
#     name, erp = None, None
#     if qrs:
#         qr_data = qrs[0].data.decode("utf-8")
#         name, erp = parse_qr_text(qr_data)

#     if qr_data and qr_data not in processed_qrs:
#         processed_qrs.add(qr_data)
#         print(f"\nQR detected: {name} | {erp}")
#         print("Place your plate in front of the camera...")
#         time.sleep(2)  # give time to place plate

#         # Capture fresh frame for plate detection
#         ret2, plate_frame = cap.read()
#         if not ret2:
#             print("Failed to capture plate frame")
#             continue

#         # ---------- YOLO SEGMENTATION ----------
#         results = model(plate_frame, conf=0.15, iou=0.5, verbose=False)

#         plate_mask = None
#         food_mask = None

#         for r in results:
#             if r.masks is None:
#                 continue
#             for cls, mask in zip(r.boxes.cls, r.masks.data):
#                 cls_id = int(cls)
#                 mask_np = mask.cpu().numpy().astype(bool)
#                 if cls_id == PLATE_CLASS_ID:
#                     plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)
#                 if cls_id in FOOD_CLASSES:
#                     food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

#         # ---------- FALLBACK: CIRCLE DETECTION ----------
#         if plate_mask is None:
#             gray = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
#             blur = cv2.GaussianBlur(gray, (9, 9), 2)
#             circles = cv2.HoughCircles(
#                 blur, cv2.HOUGH_GRADIENT,
#                 dp=1.1, minDist=200, param1=120, param2=30,
#                 minRadius=120, maxRadius=240
#             )
#             if circles is not None:
#                 circles = np.uint16(np.around(circles))
#                 x, y, r = max(circles[0], key=lambda c: c[2])
#                 plate_mask = np.zeros(plate_frame.shape[:2], dtype=np.uint8)
#                 cv2.circle(plate_mask, (x, y), r, 255, -1)
#                 plate_mask = plate_mask.astype(bool)
#                 cv2.circle(plate_frame, (x, y), r, (0, 255, 0), 3)

#         # ---------- CALCULATE WASTE ----------
#         if plate_mask is not None:
#             plate_area = np.sum(plate_mask)
#             if plate_area > 0 and food_mask is not None:
#                 food_on_plate = np.logical_and(food_mask, plate_mask)
#                 food_area = np.sum(food_on_plate)
#                 waste_pct = (food_area / plate_area) * 100
#             else:
#                 waste_pct = 0

#             # ---------- VISUALIZATION ----------
#             overlay = plate_frame.copy()
#             overlay[plate_mask] = (0, 255, 0)
#             plate_frame = cv2.addWeighted(overlay, 0.3, plate_frame, 0.7, 0)
#             if food_mask is not None and np.sum(food_on_plate) > 0:
#                 overlay2 = plate_frame.copy()
#                 overlay2[food_on_plate] = (0, 0, 255)
#                 plate_frame = cv2.addWeighted(overlay2, 0.4, plate_frame, 0.6, 0)

#             cv2.putText(
#                 plate_frame, f"Food Waste: {waste_pct:.2f}%", (30, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
#             )

#             print(f"Plate detected! Food Waste: {waste_pct:.2f}%")

#             # Save attendance + waste info
#             ts = time.localtime()
#             date = time.strftime("%Y-%m-%d", ts)
#             tstr = time.strftime("%H:%M:%S", ts)
#             with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([name, erp, date, tstr, f"{waste_pct:.2f}%"])

#             # Show the detection frame for a short time
#             cv2.imshow("Plate Food Waste", plate_frame)
#             cv2.waitKey(int(SHOW_TIME * 1000))

#         else:
#             print("Plate not detected!")
#             cv2.putText(
#                 plate_frame, "Plate not detected", (30, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
#             )
#             cv2.imshow("Plate Food Waste", plate_frame)
#             cv2.waitKey(int(SHOW_TIME * 1000))

#     # Display live camera feed
#     cv2.putText(frame, "Show QR code | Press Q to quit", (10, frame.shape[0]-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#     cv2.imshow("Plate Food Waste", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()
# print("Program ended safely.")



# much faster 


# import cv2
# import numpy as np
# import time
# import csv
# import os
# from pyzbar.pyzbar import decode
# from ultralytics import YOLO

# # ---------------- CSV ----------------
# CSV_FILE = "attendance.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time", "Food_Waste_Percentage"])

# def parse_qr_text(text):
#     """Parse QR text formatted as 'Name:John;ERP:12345'"""
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

# # ---------------- YOLO MODEL ----------------
# model = YOLO("yolov8n-seg.pt")  # Use your trained segmentation model

# PLATE_CLASS_ID = 60  # COCO plate class
# FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Camera not opening")
#     exit()

# print("Show QR code to camera. Press Q or ESC to quit.")

# # Throttle inference
# INFER_DELAY = 0.6
# last_infer_time = 0
# SHOW_TIME = 2.5  # seconds to display detection frame

# processed_qrs = set()

# # ---------------- MAIN LOOP ----------------
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # -------- QR DETECTION --------
#     qrs = decode(frame)
#     qr_data = None
#     name, erp = None, None
#     if qrs:
#         qr_data = qrs[0].data.decode("utf-8")
#         name, erp = parse_qr_text(qr_data)

#     if qr_data and qr_data not in processed_qrs:
#         processed_qrs.add(qr_data)
#         print(f"\nQR detected: {name} | {erp}")
#         print("Place your plate in front of the camera...")
#         time.sleep(2)  # give time to place plate

#         # Capture fresh frame for plate detection
#         ret2, plate_frame = cap.read()
#         if not ret2:
#             print("Failed to capture plate frame")
#             continue

#         # ---------- YOLO SEGMENTATION ----------
#         results = model(plate_frame, conf=0.15, iou=0.5, verbose=False)

#         plate_mask = None
#         food_mask = None

#         for r in results:
#             if r.masks is None:
#                 continue
#             for cls, mask in zip(r.boxes.cls, r.masks.data):
#                 cls_id = int(cls)
#                 mask_np = mask.cpu().numpy().astype(bool)
#                 if cls_id == PLATE_CLASS_ID:
#                     plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)
#                 if cls_id in FOOD_CLASSES:
#                     food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

#         # ---------- FALLBACK: CIRCLE DETECTION ----------
#         if plate_mask is None:
#             gray = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
#             blur = cv2.GaussianBlur(gray, (9, 9), 2)
#             circles = cv2.HoughCircles(
#                 blur, cv2.HOUGH_GRADIENT,
#                 dp=1.1, minDist=200, param1=120, param2=30,
#                 minRadius=120, maxRadius=240
#             )
#             if circles is not None:
#                 circles = np.uint16(np.around(circles))
#                 x, y, r = max(circles[0], key=lambda c: c[2])
#                 plate_mask = np.zeros(plate_frame.shape[:2], dtype=np.uint8)
#                 cv2.circle(plate_mask, (x, y), r, 255, -1)
#                 plate_mask = plate_mask.astype(bool)
#                 cv2.circle(plate_frame, (x, y), r, (0, 255, 0), 3)

#         # ---------- CALCULATE WASTE ----------
#         if plate_mask is not None:
#             plate_area = np.sum(plate_mask)
#             if plate_area > 0 and food_mask is not None:
#                 food_on_plate = np.logical_and(food_mask, plate_mask)
#                 food_area = np.sum(food_on_plate)
#                 waste_pct = (food_area / plate_area) * 100
#             else:
#                 waste_pct = 0

#             # ---------- VISUALIZATION ----------
#             overlay = plate_frame.copy()
#             overlay[plate_mask] = (0, 255, 0)
#             plate_frame = cv2.addWeighted(overlay, 0.3, plate_frame, 0.7, 0)
#             if food_mask is not None and np.sum(food_on_plate) > 0:
#                 overlay2 = plate_frame.copy()
#                 overlay2[food_on_plate] = (0, 0, 255)
#                 plate_frame = cv2.addWeighted(overlay2, 0.4, plate_frame, 0.6, 0)

#             cv2.putText(
#                 plate_frame, f"Food Waste: {waste_pct:.2f}%", (30, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
#             )

#             print(f"Plate detected! Food Waste: {waste_pct:.2f}%")

#             # ---------- SAVE TO CSV ----------
#             ts = time.localtime()
#             date = time.strftime("%Y-%m-%d", ts)
#             tstr = time.strftime("%H:%M:%S", ts)
#             try:
#                 with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     writer.writerow([name, erp, date, tstr, f"{waste_pct:.2f}%"])
#                     f.flush()
#                 print(f"Data saved to {CSV_FILE}")
#             except Exception as e:
#                 print("Failed to write CSV:", e)

#             # Show detection frame for a short time
#             cv2.imshow("Plate Food Waste", plate_frame)
#             cv2.waitKey(int(SHOW_TIME * 1000))

#         else:
#             print("Plate not detected!")
#             cv2.putText(
#                 plate_frame, "Plate not detected", (30, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
#             )
#             cv2.imshow("Plate Food Waste", plate_frame)
#             cv2.waitKey(int(SHOW_TIME * 1000))

#     # Display live camera feed
#     cv2.putText(frame, "Show QR code | Press Q to quit", (10, frame.shape[0]-10),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
#     cv2.imshow("Plate Food Waste", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q') or key == 27:
#         break

# # ---------------- CLEANUP ----------------
# cap.release()
# cv2.destroyAllWindows()
# print("Program ended safely.")


# real time percentage-------------------------------
# import cv2
# import numpy as np
# import time
# import csv
# import os
# from pyzbar.pyzbar import decode
# from ultralytics import YOLO




# import firebase_admin
# from firebase_admin import credentials, firestore

# cred = credentials.Certificate("firebase_key.json")
# # firebase_admin.initialize_app(cred)
# if not firebase_admin._apps:
#     firebase_admin.initialize_app(cred)

# db = firestore.client()

# print("Firebase connected")


# # ---------------- CSV ----------------
# # CSV_FILE = "attendance.csv"
# # if not os.path.exists(CSV_FILE):
# #     with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["Name", "ERP", "Date", "Time", "Food_Waste_Percentage"])

# # def parse_qr_text(text):
# #     """Parse QR text formatted as 'Name:John;ERP:12345'"""
# #     text = text.strip()
# #     if ";" in text:
# #         parts = {}
# #         for p in text.split(";"):
# #             if ":" in p:
# #                 k, v = p.split(":",1)
# #                 parts[k.strip().lower()] = v.strip()
# #         name = parts.get("name", "Unknown")
# #         erp = parts.get("erp", "Unknown")
# #         return name, erp
# #     return text, "Unknown"

# # # ---------------- YOLO MODEL ----------------
# # # Use a YOLO segmentation model trained or fine-tuned for plate+food if possible
# # model = YOLO("yolov8n-seg.pt")

# # # Only detect plates and food
# # PLATE_CLASS_ID = 60
# # FOOD_CLASSES = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}

# # # ---------------- CAMERA ----------------
# # cap = cv2.VideoCapture(0)
# # if not cap.isOpened():
# #     print("Camera not opening")
# #     exit()

# # print("Show QR code to camera. Press Q or ESC to quit.")

# # # ---------------- PARAMETERS ----------------
# # INFER_DELAY = 0.4  # seconds
# # last_infer_time = 0
# # SHOW_TIME = 1.5  # seconds to display plate detection frame
# # processed_qrs = set()

# # # ---------------- MAIN LOOP ----------------
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break

# #     # -------- QR DETECTION --------
# #     qrs = decode(frame)
# #     qr_data = None
# #     name, erp = None, None
# #     if qrs:
# #         qr_data = qrs[0].data.decode("utf-8")
# #         name, erp = parse_qr_text(qr_data)

# #     # If a new QR detected
# #     if qr_data and qr_data not in processed_qrs:
# #         processed_qrs.add(qr_data)
# #         print(f"\nQR detected: {name} | {erp}")
# #         print("Place your plate in front of the camera...")
# #         time.sleep(1.5)  # short wait for plate

# #         # Capture fresh frame for plate detection
# #         ret2, plate_frame = cap.read()
# #         if not ret2:
# #             print("Failed to capture plate frame")
# #             continue

# #         # ---------- YOLO SEGMENTATION ----------
# #         results = model(plate_frame, conf=0.2, iou=0.4, verbose=False, classes=[PLATE_CLASS_ID]+list(FOOD_CLASSES))

# #         plate_mask = None
# #         food_mask = None

# #         for r in results:
# #             if r.masks is None:
# #                 continue
# #             for cls, mask in zip(r.boxes.cls, r.masks.data):
# #                 cls_id = int(cls)
# #                 mask_np = mask.cpu().numpy().astype(bool)
# #                 if cls_id == PLATE_CLASS_ID:
# #                     plate_mask = mask_np if plate_mask is None else np.logical_or(plate_mask, mask_np)
# #                 if cls_id in FOOD_CLASSES:
# #                     food_mask = mask_np if food_mask is None else np.logical_or(food_mask, mask_np)

# #         # ---------- FALLBACK: CIRCLE DETECTION ----------
# #         if plate_mask is None:
# #             gray = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
# #             blur = cv2.GaussianBlur(gray, (9, 9), 2)
# #             circles = cv2.HoughCircles(
# #                 blur, cv2.HOUGH_GRADIENT,
# #                 dp=1.1, minDist=200, param1=120, param2=30,
# #                 minRadius=120, maxRadius=240
# #             )
# #             if circles is not None:
# #                 circles = np.uint16(np.around(circles))
# #                 x, y, r = max(circles[0], key=lambda c: c[2])
# #                 plate_mask = np.zeros(plate_frame.shape[:2], dtype=np.uint8)
# #                 cv2.circle(plate_mask, (x, y), r, 255, -1)
# #                 plate_mask = plate_mask.astype(bool)
# #                 cv2.circle(plate_frame, (x, y), r, (0, 255, 0), 3)

# #         # ---------- CALCULATE WASTE ----------
# #         if plate_mask is not None:
# #             plate_area = np.sum(plate_mask)
# #             if plate_area > 0 and food_mask is not None:
# #                 food_on_plate = np.logical_and(food_mask, plate_mask)
# #                 food_area = np.sum(food_on_plate)
# #                 waste_pct = (food_area / plate_area) * 100
# #             else:
# #                 waste_pct = 0

# #             # ---------- VISUALIZATION ----------
# #             overlay = plate_frame.copy()
# #             overlay[plate_mask] = (0, 255, 0)
# #             plate_frame = cv2.addWeighted(overlay, 0.3, plate_frame, 0.7, 0)
# #             if food_mask is not None and np.sum(food_on_plate) > 0:
# #                 overlay2 = plate_frame.copy()
# #                 overlay2[food_on_plate] = (0, 0, 255)
# #                 plate_frame = cv2.addWeighted(overlay2, 0.4, plate_frame, 0.6, 0)

# #             cv2.putText(
# #                 plate_frame, f"Food Waste: {waste_pct:.2f}%", (30, 40),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
# #             )

# #             print(f"Plate detected! Food Waste: {waste_pct:.2f}%")

# #             # ---------- SAVE TO CSV ----------
# #             ts = time.localtime()
# #             date = time.strftime("%Y-%m-%d", ts)
# #             tstr = time.strftime("%H:%M:%S", ts)
# #             try:
# #                 with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
# #                     writer = csv.writer(f)
# #                     writer.writerow([name, erp, date, tstr, f"{waste_pct:.2f}%"])
# #                     f.flush()
# #                 print(f"Data saved to {CSV_FILE}")
# #             except Exception as e:
# #                 print("Failed to write CSV:", e)

# #             # Show plate frame briefly
# #             cv2.imshow("Plate Food Waste", plate_frame)
# #             cv2.waitKey(int(SHOW_TIME*1000))
# #         else:
# #             print("Plate not detected!")
# #             cv2.putText(
# #                 plate_frame, "Plate not detected", (30, 40),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
# #             )
# #             cv2.imshow("Plate Food Waste", plate_frame)
# #             cv2.waitKey(int(SHOW_TIME*1000))

# #     # Display live camera feed
# #     cv2.putText(frame, "Show QR code | Press Q to quit", (10, frame.shape[0]-10),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
# #     cv2.imshow("Plate Food Waste", frame)

# #     key = cv2.waitKey(1) & 0xFF
# #     if key == ord('q') or key == 27:
# #         break

# # # ---------------- CLEANUP ----------------
# # cap.release()
# # cv2.destroyAllWindows()
# # print("Program ended safely.")




# import cv2
# import numpy as np
# import time
# import firebase_admin
# from firebase_admin import credentials, firestore
# from pyzbar.pyzbar import decode
# from ultralytics import YOLO

# # ---------------- FIREBASE ----------------
# cred = credentials.Certificate("firebase_key.json")
# if not firebase_admin._apps:
#     firebase_admin.initialize_app(cred)
# db = firestore.client()
# print("✅ Firebase connected")

# # ---------------- YOLO MODEL ----------------
# model = YOLO("yolov8n-seg.pt")  # YOLO segmentation model
# PLATE_CLASS_ID = 60  # Only detect plate class

# # ---------------- CAMERA ----------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ Camera not opening")
#     exit()

# processed_qrs = set()
# print("Show QR code to camera...")

# # ---------------- MAIN LOOP ----------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     qr_data = None
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # ---------- QR SCAN ----------
#     for qr in decode(gray):
#         qr_data = qr.data.decode("utf-8")
#         x, y, w, h = qr.rect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, qr_data, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     cv2.imshow("Plate Food Detection", frame)

#     # ---------- PROCESS QR ONCE ----------
#     if qr_data and qr_data not in processed_qrs:
#         processed_qrs.add(qr_data)
#         print(f"✅ Student Identified: {qr_data}")

#         # wait 4 seconds for plate positioning
#         time.sleep(4)

#         # ---------- YOLO PLATE & FOOD DETECTION ----------
#         ret2, plate_frame = cap.read()  # capture current frame for food detection
#         if not ret2:
#             continue

#         results = model(plate_frame)  # segmentation
#         food_area = 0
#         plate_area = 0

#         if results:
#             for r in results:
#                 if r.masks is not None:
#                     for mask, cls in zip(r.masks.data, r.boxes.cls):
#                         cls_id = int(cls)
#                         mask_area = mask.sum().item()
#                         mask_img = mask.cpu().numpy().astype(np.uint8) * 255

#                         if cls_id == PLATE_CLASS_ID:
#                             plate_area += mask_area
#                             # Draw circle around plate
#                             contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                             for cnt in contours:
#                                 (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
#                                 cv2.circle(plate_frame, (int(x_c), int(y_c)), int(radius), (255, 0, 0), 2)
#                         else:
#                             food_area += mask_area
#                             # Overlay green mask for food
#                             mask_color = cv2.merge([np.zeros_like(mask_img), mask_img, np.zeros_like(mask_img)])
#                             plate_frame = cv2.addWeighted(plate_frame, 1, mask_color, 0.5, 0)

#         food_percent = 0
#         if plate_area > 0:
#             food_percent = int((food_area / plate_area) * 100)

#         cv2.putText(plate_frame, f"Food Left: {food_percent}%", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Plate Food Detection", plate_frame)
#         cv2.waitKey(1500)  # show the processed frame for 1.5 sec

#         # ---------- SAVE TO FIREBASE ----------
#         try:
#             student_info = qr_data.split(",")  # QR = "Name,ERP"
#             name = student_info[0]
#             erp = student_info[1] if len(student_info) > 1 else "Unknown"
#             timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

#             db.collection("food_waste").add({
#                 "Name": name,
#                 "ERP": erp,
#                 "Date": timestamp.split(" ")[0],
#                 "Time": timestamp.split(" ")[1],
#                 "Food_Waste_Percentage": food_percent
#             })

#             print(f"✅ Data saved for {name}, Food Left: {food_percent}%")
#         except:
#             print("⚠️ QR format wrong. Use: Name,ERP")

#     # Press Q to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()







import cv2
import time
import numpy as np
import csv
import firebase_admin
from firebase_admin import credentials, firestore
from pyzbar.pyzbar import decode
import os

# ================= CONFIG =================
FIREBASE_JSON = "firebase_key.json"
CSV_FILE = "food_waste_backup.csv"
QR_WAIT = 4                 # seconds after QR
PLATE_RADIUS_PIXELS = 150   # adjust once and keep camera fixed

# ================= FIREBASE =================
if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_JSON)
    firebase_admin.initialize_app(cred)

db = firestore.client()
print("✅ Firebase connected")

# ================= CSV BACKUP =================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "ERP", "Date", "Time", "Food_Waste_Percentage"])

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

processed_qrs = set()
print("📷 Show QR code to camera...")

# ================= PLATE DETECTION =================
def detect_plate(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=120,
        param1=100,
        param2=30,
        minRadius=int(PLATE_RADIUS_PIXELS * 0.9),
        maxRadius=int(PLATE_RADIUS_PIXELS * 1.1)
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return (x, y), r

    return None, None

# ================= FOOD DETECTION (STRICT) =================
def process_plate(frame, center, radius):
    overlay = frame.copy()

    # Plate mask
    plate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(plate_mask, center, radius, 255, -1)

    # Extract only plate
    plate_only = cv2.bitwise_and(frame, frame, mask=plate_mask)

    # HSV conversion
    hsv = cv2.cvtColor(plate_only, cv2.COLOR_BGR2HSV)

    # FOOD RULES:
    # ✔ medium saturation
    # ✔ medium brightness
    # ✖ steel = low saturation + high brightness
    food_mask = cv2.inRange(
        hsv,
        np.array([5, 70, 40]),     # lower
        np.array([170, 255, 200]) # upper
    )

    # Remove steel reflections
    reflection = cv2.inRange(
        hsv,
        np.array([0, 0, 200]),
        np.array([179, 60, 255])
    )
    food_mask = cv2.subtract(food_mask, reflection)

    # Clean mask
    kernel = np.ones((5, 5), np.uint8)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_OPEN, kernel)
    food_mask = cv2.morphologyEx(food_mask, cv2.MORPH_CLOSE, kernel)

    food_inside = cv2.bitwise_and(food_mask, plate_mask)

    # Calculate percentage
    plate_area = cv2.countNonZero(plate_mask)
    food_area = cv2.countNonZero(food_inside)
    food_percent = int((food_area / plate_area) * 100) if plate_area else 0

    # Visualization
    cv2.circle(overlay, center, radius, (255, 0, 0), 2)
    green = np.zeros_like(frame)
    green[:, :, 1] = food_inside
    overlay = cv2.addWeighted(overlay, 1, green, 0.6, 0)

    cv2.putText(
        overlay,
        f"Food Waste: {food_percent}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    return overlay, food_percent

# ================= SAVE DATA =================
def save_data(name, erp, food_percent):
    date = time.strftime("%Y-%m-%d")
    time_now = time.strftime("%H:%M:%S")

    # Firebase
    db.collection("food_waste").add({
        "Name": name,
        "ERP": erp,
        "Date": date,
        "Time": time_now,
        "Food_Waste_Percentage": food_percent
    })

    # CSV backup
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, erp, date, time_now, food_percent])

    print(f"✅ Saved → {name} | Food Waste: {food_percent}%")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    qr_data = None

    # ---------- QR SCAN ----------
    for qr in decode(gray):
        qr_data = qr.data.decode("utf-8")
        x, y, w, h = qr.rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, qr_data, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("QR Detection", frame)

    # ---------- PROCESS ONCE PER QR ----------
    if qr_data and qr_data not in processed_qrs:
        processed_qrs.add(qr_data)
        print(f"🧾 QR detected: {qr_data}")
        time.sleep(QR_WAIT)

        ret2, plate_frame = cap.read()
        if not ret2:
            continue

        gray_plate = cv2.cvtColor(plate_frame, cv2.COLOR_BGR2GRAY)
        center, radius = detect_plate(gray_plate)

        if center is None:
            print("⚠️ Plate not detected properly")
            continue

        result, food_percent = process_plate(plate_frame, center, radius)
        cv2.imshow("Food Waste Result", result)
        cv2.waitKey(1500)

        try:
            data = qr_data.split(",")
            name = data[0]
            erp = data[1] if len(data) > 1 else "Unknown"
            save_data(name, erp, food_percent)
        except:
            print("⚠️ QR format must be: Name,ERP")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()