import cv2
from pyzbar.pyzbar import decode

def scan_qr(frame):
    data = None
    for qr in decode(frame):
        data = qr.data.decode("utf-8")
        x, y, w, h = qr.rect
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, data, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        break
    return data, frame


