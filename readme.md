# 🍽️ Smart Food Waste Detection System

## Hello and Welcome! 👋

Have you ever noticed how much food gets wasted in our college cafeteria?  
We did, and we wanted to do something about it. That’s why we built the **Smart Food Waste Detection System**!  

This system can **scan a student’s QR code**, detect their plate, and calculate **exactly how much food is left**. All results are saved in **Firebase** and a local **CSV backup**.  

It’s simple, fast, and helps our campus become more **eco-friendly**. 🌱

---

## 🚀 Features

- **QR Code Scanning** – Instantly identifies student name and ERP  
- **Plate Detection** – Uses optimized detection to locate the plate accurately  
- **Food Percentage Calculation** – Only detects food inside the plate; ignores background or people  
- **Live Feedback** – Green circle overlays show plate detection in real-time  
- **Data Storage** – Saves results automatically in Firebase and a CSV file  
- **Terminal Logging** – Displays real-time information for each student  

---

## 🛠️ What You Need

- Python 3.10+  
- OpenCV  
- Numpy  
- Pyzbar  
- Firebase Admin SDK  

Install the required libraries easily with:

```bash
pip install opencv-python numpy pyzbar firebase-admin
