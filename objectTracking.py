
#---------------------------------------------------#
# Script 1:

import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

# Tracking with Bot-Sort (default tracker)
# results = model.track (source = "videos/video8.mp4", show=True, save=True, tracker = "botsort.yaml") 

# Tracking with Byte-Track
# results = model.track (source = "videos/video8.mp4", save=True, tracker = "bytetrack.yaml", conf = 0.20, iou = 0.3)

# exit(0)

#---------------------------------------------------#
# Script 2:

# Python Script using OpenCV-Python (cv2) and YOLO11 to run Object Tracking on Video Frames or on Live Webcam Feed

import cv2
from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

# Create a Video Capture Object
cap = cv2.VideoCapture ("videos/video5.mp4")

# Loop through Video Frames
while True:
    ret, frame = cap.read()
    if not ret: break
        
    # Run YOLO11 Tracking on the Video Frames
    results = model.track(frame, persist=True)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the annotated frame
    cv2.imshow("YOLO11 Tracking", annotated_frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

