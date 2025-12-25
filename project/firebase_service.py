import firebase_admin
from firebase_admin import credentials, firestore
import os
import datetime

def init_firebase():
    if not firebase_admin._apps:
        # Get path of THIS file
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Firebase key must be inside project folder
        key_path = os.path.join(base_dir, "firebase_key.json")

        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Firebase key not found at: {key_path}")

        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)

    return firestore.client()

def save_data(db, erp, waste):
    try:
        data = {
            "erp": erp,
            "waste_percent": waste,
            "timestamp": datetime.datetime.now()
        }
        db.collection("food_waste").add(data)
        print("✅ Saved to Firebase")
    except Exception as e:
        print("❌ Firebase save error:", e)





# import firebase_admin
# from firebase_admin import credentials, firestore
# import os

# def init_firebase():
#     if not firebase_admin._apps:
#         cred = credentials.Certificate("firebase_key.json")
#         firebase_admin.initialize_app(cred)
#     return firestore.client()

# def save_data(db, erp, waste):
#     try:
#         data = {
#             "erp": erp,
#             "waste_percentage": int(waste),
#             "timestamp": firestore.SERVER_TIMESTAMP
#         }

#         db.collection("food_waste_logs").add(data)
#         print("🔥 Data saved to Firebase successfully")

#     except Exception as e:
#         print("❌ Firebase save error:", e)







