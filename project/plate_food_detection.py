import cv2
from ultralytics import YOLO
import pandas as pd
import qrcode
import os
import datetime

# -------------------------
# 1. Setup
# -------------------------
# Load YOLO model for plate + food detection
model = YOLO("yolov8n.pt")  # use custom trained for plate + food if available

# CSV file to save attendance + waste
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Name", "ERP", "Date", "Time", "Waste"])
    df.to_csv(csv_file, index=False)

# Folder to save QR codes (optional)
qr_folder = "QR_Students"
if not os.path.exists(qr_folder):
    os.makedirs(qr_folder)

# Open camera
cap = cv2.VideoCapture(0)

print("Press 'q' to quit anytime.")

# -------------------------
# 2. Loop to process students
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------- QR Code Detection --------
    qr_detector = cv2.QRCodeDetector()
    data, bbox, _ = qr_detector.detectAndDecode(frame)

    student_name = None
    student_erp = None

    if data:
        # Example QR format: "Name:Priyanshu;ERP:12345"
        try:
            parts = data.split(";")
            student_name = parts[0].split(":")[1]
            student_erp = parts[1].split(":")[1]

            # Generate QR code image for record (optional)
            qr_img = qrcode.make(data)
            qr_img.save(os.path.join(qr_folder, f"{student_erp}.png"))

            print(f"QR Detected: {student_name} | ERP: {student_erp}")
        except:
            student_name = "Unknown"
            student_erp = "Unknown"

    # -------- Plate + Food Detection --------
    results = model(frame)[0]

    plate_detected = False
    food_area = 0
    plate_area = 0
    plate_box = None

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Adjust class IDs based on your model
        if cls == 0 and conf > 0.5:  # plate
            plate_detected = True
            plate_area = (x2 - x1) * (y2 - y1)
            plate_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        elif cls in [1,2,3] and conf > 0.5:  # food items
            food_area += (x2 - x1) * (y2 - y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    # Waste calculation
    if plate_detected and plate_area > 0:
        ratio = food_area / plate_area
        if ratio > 0.5:
            status = "HIGH"
            color = (0,0,255)
        elif ratio > 0.2:
            status = "MEDIUM"
            color = (0,165,255)
        elif ratio > 0.05:
            status = "LOW"
            color = (0,255,0)
        else:
            status = "EMPTY"
            color = (255,255,255)
    else:
        status = "NO PLATE"
        color = (0,0,255)

    # Show status on frame
    cv2.putText(frame, f"Waste: {status}", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Attendance + Food Waste", frame)

    # -------- Save attendance after QR + plate detection --------
    if student_name and plate_detected:
        now = datetime.datetime.now()
        df = pd.read_csv(csv_file)
        # Avoid duplicates for same ERP today
        if not ((df["ERP"] == student_erp) & (df["Date"] == now.date().strftime("%Y-%m-%d"))).any():
            new_entry = {
                "Name": student_name,
                "ERP": student_erp,
                "Date": now.date().strftime("%Y-%m-%d"),
                "Time": now.time().strftime("%H:%M:%S"),
                "Waste": status
            }
            df = df.append(new_entry, ignore_index=True)
            df.to_csv(csv_file, index=False)
            print(f"Saved: {student_name} | Waste: {status}")

    # Exit loop safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()