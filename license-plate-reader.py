# Import All the Required Libraries
import cv2
import math
import time
import easyocr
from ultralytics import YOLO

# NOTES:   (12:04)
#     - There is an issue with easyocr. It installs a headless version of opencv-python which causes cv2.imshow to not work.
#     - Install "easyocr" first, then install "ultralytics" (which installs "opencv-python") to fix the issue.

# Create a Video Capture Object
cap = cv2.VideoCapture("videos/4.mp4")

# Class Name
classNameFT = ["Licence Plate"]

# Save the Output Video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter(
    filename = 'output-plate-reader-2.avi', 
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
    fps = fps, 
    frameSize = (frame_width, frame_height))

# FPS Calcs
ptime = 0
ctime = 0
count = 0

# Load the Fine-Tune Model
model = YOLO("weights/best.pt")

# Read the Text from License Plate using EasyOCR
reader = easyocr.Reader(lang_list = ['en'], gpu = False)

def ocr_image (frame, x1, y1, x2, y2):
    frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray)
    # print(result)
    text = ""
    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2: # confidence > 0.2
            text = res[1]
    return str(text)

# Loop through the Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        count += 1
        print(f"Frame Count: {count}")
        results = model.predict(source = frame, conf = 0.25, iou = 0.2)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0] # tensor
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cls = int(box.cls[0])
                className = classNameFT[cls]
                conf = math.ceil(box.conf[0] * 100)/100
                label = ocr_image(frame, x1, y1, x2, y2)   # OCR
                # label = f"{className}:{conf}"  # Not OCR
                textSize = cv2.getTextSize(label, 0, fontScale = 0.5, thickness = 2)[0]
                c2 = x1 + textSize[0], y1 - textSize[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
        ctime = time.time() 
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(frame, 'FPS' + ":" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        output.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

# Release the memory
cap.release()
cv2.destroyAllWindows()
