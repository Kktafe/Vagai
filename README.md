Here's the recommended file and folder structure for this project:

```
pulse_sorting_system/
│
├── vAgAI.py                          # Main application code (rename from .txt)
│
├── models/
│   ├── yolov8n.pt                    # Pre-trained base model (download first time)
│   └── best.pt                       # Your custom-trained model (after training)
│
├── dataset/                          # Training data (needed for model training)
│   ├── images/
│   │   ├── train/                    # Training images
│   │   │   ├── img_001.jpg
│   │   │   ├── img_002.jpg
│   │   │   └── ...
│   │   └── val/                      # Validation images
│   │       ├── img_val_001.jpg
│   │       └── ...
│   ├── labels/
│   │   ├── train/                    # YOLO format annotations
│   │   │   ├── img_001.txt
│   │   │   ├── img_002.txt
│   │   │   └── ...
│   │   └── val/
│   │       ├── img_val_001.txt
│   │       └── ...
│   └── dataset.yaml                  # Dataset configuration file
│
├── rejects/                          # Auto-created folder for rejected item images
│   ├── pulse_123_cam_0.jpg          # (Created automatically during runtime)
│   ├── pulse_456_cam_1.jpg
│   └── ...
│
├── logs/
│   └── reject_log.csv                # Rejection event log (auto-created)
│
├── config/
│   └── settings.json                 # Optional: Store thresholds, camera settings
│
├── runs/                             # YOLOv8 training outputs (auto-created)
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── .gitignore                        # Ignore large files, logs, etc.
```

## Essential Files to Create Before Running:

### 1. **requirements.txt**
```txt
opencv-python==4.8.1.78
streamlit==1.28.0
pandas==2.1.0
ultralytics==8.0.200
numpy==1.24.3
```

### 2. **dataset.yaml** (in dataset folder)
```yaml
path: ../dataset  # Relative to training script
train: images/train
val: images/val

nc: 1  # Number of classes
names: ['pulse']  # Class names
```

### 3. **config/settings.json** (optional but recommended)
```json
{
  "model_path": "models/best.pt",
  "camera_ids": [0, 1, 2],
  "width_threshold": {
    "min": 10,
    "max": 50
  },
  "color_range_hsv": {
    "lower": [0, 100, 100],
    "upper": [10, 255, 255]
  },
  "rejection_delay_ms": 500,
  "log_path": "logs/reject_log.csv"
}
```

### 4. **.gitignore**
```
# Datasets
dataset/images/
dataset/labels/

# Model weights
*.pt
models/*.pt
runs/

# Runtime data
rejects/
logs/*.csv

# Python
__pycache__/
*.pyc
.env

# Streamlit
.streamlit/
```

## Code Modifications Required:

Update the paths in your main code:

```python
# --- CONFIGURATION ---
MODEL_PATH = 'models/best.pt'  # Changed from 'yolov8n.pt'
CAM_IDS = [0, 1, 2]
WIDTH_THRESHOLD = (10, 50)
COLOR_RANGE = ([0, 100, 100], [10, 255, 255])

# Ensure folders exist
import os
os.makedirs("rejects", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Updated log path
def handle_rejection(pulse_id, cam_id, reason, frame):
    log_entry = f"{time.time()},{pulse_id},{cam_id},{reason}\n"
    with open("logs/reject_log.csv", "a") as f:  # Updated path
        f.write(log_entry)
    
    cv2.imwrite(f"rejects/pulse_{pulse_id}_cam_{cam_id}.jpg", frame)
```

## Setup Steps:

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create necessary folders**:
   ```bash
   mkdir -p models rejects logs dataset/images/train dataset/images/val dataset/labels/train dataset/labels/val
   ```

3. **Download base model** (first run will auto-download):
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Auto-downloads to local cache
   ```

4. **Run the application**:
   ```bash
   streamlit run vAgAI.py
   ```

This structure keeps everything organized, separates training data from runtime data, and makes the project easy to maintain and deploy.
