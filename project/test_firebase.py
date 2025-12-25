import numpy as np
import cv2

def create_plate_roi(frame, plate_diameter_cm=30, camera_fov_cm=50):
    """
    Creates a circular ROI mask for the plate at the center.
    
    Returns:
        mask: binary mask of plate region
        radius_px: radius in pixels
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    frame_width_px = frame.shape[1]
    radius_px = int((plate_diameter_cm / camera_fov_cm) * frame_width_px / 2)
    center = (frame.shape[1]//2, frame.shape[0]//2)
    cv2.circle(mask, center, radius_px, 1, -1)
    return mask, radius_px

def draw_plate_roi(frame, radius_px):
    """
    Draws the ROI circle on the frame for guidance
    """
    center = (frame.shape[1]//2, frame.shape[0]//2)
    cv2.circle(frame, center, radius_px, (0,255,0), 2)
    cv2.putText(frame, f"Plate ROI: {radius_px}px", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
