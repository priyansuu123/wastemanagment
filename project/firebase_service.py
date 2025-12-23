import csv
import os
import time
import firebase_admin
from firebase_admin import credentials, firestore
from config import FIREBASE_JSON, CSV_FILE

def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_JSON)
        firebase_admin.initialize_app(cred)
    print("✅ Firebase connected")
    return firestore.client()

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "ERP", "Date", "Time", "Food_Waste_Percentage"])
        print("CSV backup file created")
    else:
        print("CSV backup file found")

def save_data(db, name, erp, food_percent):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    db.collection("food_waste").add({
        "Name": name,
        "ERP": erp,
        "Date": timestamp.split(" ")[0],
        "Time": timestamp.split(" ")[1],
        "Food_Waste_Percentage": food_percent
    })

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            erp,
            timestamp.split(" ")[0],
            timestamp.split(" ")[1],
            food_percent
        ])

    print(f"✅ Data saved | {name} | {food_percent}%")
