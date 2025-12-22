import qrcode
import os

# Create folder to store QR codes
folder_name = "student_qr"
os.makedirs(folder_name, exist_ok=True)

# Student data (Name, ERP)
students = [
    ("Priyanshu", "6606001"),
    ("Pallavi", "6606002"),
    ("Aduit", "6606003"),
    ("Piyush", "6606004"),
    ("Abdullah", "6606005"),
    ("Devendra", "6606006"),
    ("Shreyas", "6606007"),
    ("Rahul", "6606008"),
    ("Aman", "6606009"),
    ("Rohit", "6606010"),
    ("Ankit", "6606011"),
    ("Saurabh", "6606012"),
    ("Kunal", "6606013"),
    ("Aditya", "6606014"),
    ("Neeraj", "6606015"),
    ("Nikhil", "6606016"),
    ("Shubham", "6606017"),
    ("Deepak", "6606018"),
    ("Abhishek", "6606019"),
    # ("Manish", "6606020"),
    ("krishna","6606004"),
]

# Generate QR codes
for name, erp in students:
    qr_data = f"Name:{name},ERP:{erp}"

    qr = qrcode.make(qr_data)
    filename = f"{folder_name}/{name}_{erp}.png"
    qr.save(filename)

    print(f"QR created -> {name} | {erp}")

print("\n✅ All 20 student QR codes generated successfully!")
