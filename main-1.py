
# -----------------------------
# Steps:
# Read an image using OpenCV
# Load the YOLO11 and perform Object Detection.
# Add the Confidence Value 'conf'
# Add the classes parameter 'classes'
# Add the maximum deetction parameter 'max_det'
# Add NMS IOU 'iou'
# Add show image parameter 'show = True'
# Add 'save_txt = True', save detection results in a text file
# Add 'save_crop = True' parameter
# Object Detection on Image
# Object Detection on Video and FPS Calculation
# -----------------------------

# Import the Required Libraries
import cv2
import math
import time
from ultralytics import YOLO

# Load the YOLO11 Model
model = YOLO("yolo11n.pt")

#Read an Image using OpenCV
#image = cv2.imread("Resources/Images/image1.jpg")
#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video5.mp4")
#For Webcam
#cap = cv2.VideoCapture(0)
cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

#Object Detection using YOLO11
#results = model(image, conf = 0.25, save=False)
ptime = 0
ctime = 0
while True:
    ret, frame = cap.read()
    if ret:
        results = model(frame, conf=0.25, save=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                #Convert the Tensor into integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"x1:{x1}, y1: {y1}, x2:{x2}, y2:{y2}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), [255,0,0], 2)
                classNameInt = int(box.cls[0])
                classname = cocoClassNames[classNameInt]
                conf = math.ceil(box.conf[0] * 100)/100
                label  = classname + ":" + str(conf)
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255,0,0], -1)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType=cv2.LINE_AA)

        #Display the Video using OpenCV
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(frame, "FPS" + ":" + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()