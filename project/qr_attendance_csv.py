# import cv2
# from pyzbar.pyzbar import decode
# import csv
# from datetime import datetime

# # Keep track of scanned QR codes
# seen_qr = set()

# # CSV file to store attendance
# csv_file = "attendance.csv"

# # Create CSV with headers if it doesn't exist
# try:
#     with open(csv_file, "x", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Name", "ERP", "Date", "Time"])
# except FileExistsError:
#     pass  # File already exists

# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     for qr in decode(frame):
#         data = qr.data.decode("utf-8")  # Format: Name,ERP
#         if data not in seen_qr:
#             try:
#                 name, erp = data.split(",")
#             except ValueError:
#                 print(f"Invalid QR format: {data}")
#                 continue

#             date = datetime.now().strftime("%Y-%m-%d")
#             time = datetime.now().strftime("%H:%M:%S")

#             # Save attendance to CSV
#             with open(csv_file, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([name, erp, date, time])

#             print(f"Attendance recorded: {name} - {erp}")
#             seen_qr.add(data)

#         # Draw rectangle around QR and show name
#         x, y, w, h = qr.rect
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(frame, f"{name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     cv2.imshow("QR Scanner", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()