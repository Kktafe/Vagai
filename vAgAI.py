import cv2
import threading
import time
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'yolov8n.pt'  # Trained on your pulse dataset
CAM_IDS = [0, 1, 2]        # Top, Left, Right
WIDTH_THRESHOLD = (10, 50) # Example ranges
COLOR_RANGE = ([0, 100, 100], [10, 255, 255]) # Example HSV for reject color

# Global State for Counters
if 'counts' not in st.session_state:
    st.session_state.counts = {"pass": 0, "color_reject": 0, "size_reject": 0}

# --- ASYNC WORKERS ---
def handle_rejection(pulse_id, cam_id, reason, frame):
    """Logs data and saves images without blocking the main loop."""
    # 1. Log to CSV
    log_entry = f"{time.time()},{pulse_id},{cam_id},{reason}\n"
    with open("reject_log.csv", "a") as f:
        f.write(log_entry)
    
    # 2. Save Image
    cv2.imwrite(f"rejects/pulse_{pulse_id}_cam_{cam_id}.jpg", frame)
    
    # 3. Trigger LED (Simulated Serial Write)
    # serial_port.write(b'1') 

# --- CORE PROCESSING ---
class PulseTracker:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.track_history = deque(maxlen=30)

    def process_frame(self, frame, cam_id):
        # 1. Inference & Tracking
        results = self.model.track(frame, persist=True, verbose=False)[0]
        reject_triggered = False
        
        if results.boxes.id is not None:
            boxes = results.boxes.xywh.cpu().numpy()
            track_ids = results.boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                
                # 2. Size Validation
                if not (WIDTH_THRESHOLD[0] < w < WIDTH_THRESHOLD[1]):
                    threading.Thread(target=handle_rejection, args=(track_id, cam_id, "size", frame)).start()
                    reject_triggered = True
                
                # 3. Color Validation (Sample center of box)
                roi = frame[int(y-h/4):int(y+h/4), int(x-w/4):int(x+w/4)]
                avg_color = cv2.mean(roi)[:3] # Simplified color check
                # Logic: if avg_color not in range -> reject_triggered = True

        return results.plot(), reject_triggered

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Chute Sorting Real-Time Monitor")

col1, col2, col3, col_stats = st.columns([2, 2, 2, 1])

with col_stats:
    st.subheader("Live Stats")
    pass_metric = st.metric("Pass", st.session_state.counts["pass"])
    color_metric = st.metric("Color Rejects", st.session_state.counts["color_reject"])
    size_metric = st.metric("Size Rejects", st.session_state.counts["size_reject"])

# Initialize Cams
caps = [cv2.VideoCapture(i) for i in CAM_IDS]
trackers = [PulseTracker() for _ in CAM_IDS]
placeholders = [col1.empty(), col2.empty(), col3.empty()]

while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if ret:
            # Process frame
            processed_frame, rejected = trackers[i].process_frame(frame, i)
            
            # Update UI
            placeholders[i].image(processed_frame, channels="BGR", caption=f"Cam {i}")
            
            if rejected:
                st.session_state.counts["color_reject"] += 1 # Example logic
    
    time.sleep(0.01) # Small break for Streamlit to sync