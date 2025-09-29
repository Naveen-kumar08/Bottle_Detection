import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source, e.g., image file, video file, usb camera index, picamera', default="0")
parser.add_argument('--thresh', help='Confidence threshold', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH to display results', default=None)
parser.add_argument('--record', help='Record results to demo1.avi', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# -----------------------------
# Check model file
# -----------------------------
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or not found.')
    sys.exit(0)

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO(model_path, task='detect')
labels = model.names

# -----------------------------
# Excel + image save setup
# -----------------------------
save_folder = r"D:\project\yolo_project\captured"   # Save cropped bottles
os.makedirs(save_folder, exist_ok=True)

excel_file = r"D:\project\yolo_project\results\detections.xlsx"
os.makedirs(os.path.dirname(excel_file), exist_ok=True)

if os.path.exists(excel_file):
    df = pd.read_excel(excel_file)
else:
    df = pd.DataFrame(columns=["Bottle_ID", "Image", "Class", "Confidence", "Timestamp"])

# Bottle counter (unique ID)
bottle_id = df["Bottle_ID"].max() + 1 if not df.empty else 1

# -----------------------------
# Determine source type
# -----------------------------
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.isdigit() or 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source.replace('usb',''))
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Using laptop camera by default.')
    source_type = 'usb'
    usb_idx = 0

# -----------------------------
# Parse resolution
# -----------------------------
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# -----------------------------
# Setup recording
# -----------------------------
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video or camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# -----------------------------
# Initialize source
# -----------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video','usb']:
    cap_arg = img_source if source_type=='video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW,resH)}))
    cap.start()

# -----------------------------
# Bounding box colors
# -----------------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# -----------------------------
# Loop variables
# -----------------------------
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_idx = 0

# -----------------------------
# Main loop
# -----------------------------
while True:
    t_start = time.perf_counter()

    if source_type in ['image','folder']:
        if img_idx >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_idx])
        img_idx += 1
    elif source_type in ['video','usb']:
        ret, frame = cap.read()
        if not ret or frame is None:
            print('Camera/video not working. Exiting.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Picamera not working. Exiting.')
            break

    if resize:
        frame = cv2.resize(frame, (resW,resH))

    # Run YOLO inference
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(det.cls.item())
        classname = labels[classidx]
        conf = det.conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            object_count += 1

        # -----------------------------
        # Save bottle detections
        # -----------------------------
        if classname.lower() == "bottle":
            ymin_c, ymax_c = max(0,ymin), min(frame.shape[0],ymax)
            xmin_c, xmax_c = max(0,xmin), min(frame.shape[1],xmax)
            crop = frame[ymin_c:ymax_c, xmin_c:xmax_c]

            if crop.size == 0:
                print("Warning: Empty crop skipped")
                continue

            filename = f"bottle_{bottle_id}.jpg"
            filepath = os.path.join(save_folder, filename)
            cv2.imwrite(filepath, crop)
            print(f"[INFO] Saved bottle image: {filepath}")

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_data = {
                "Bottle_ID": bottle_id,
                "Image": filename,
                "Class": classname,
                "Confidence": round(conf,2),
                "Timestamp": timestamp
            }
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            df.to_excel(excel_file, index=False)

            bottle_id += 1

    # Overlay info
    if source_type in ['video','usb','picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5 if source_type in ['video','usb','picamera'] else 0)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)
        print("[INFO] Saved manual capture: capture.png")

    # FPS calculation
    t_stop = time.perf_counter()
    frame_rate_calc = 1/(t_stop - t_start)
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# -----------------------------
# Clean up
# -----------------------------
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type in ['video','usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()
