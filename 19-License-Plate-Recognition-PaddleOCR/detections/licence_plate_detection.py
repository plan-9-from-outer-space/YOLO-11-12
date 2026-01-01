
# Import the Required Libraries
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicencePlateDetection:

    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')  # PaddleOCR instance
    
    def detect_frames(self, frames):
        licence_plate_detections = []
        licence_plate_texts = []
        for frame in frames:
            bbox_list, text_list = self.detect_frame(frame)
            licence_plate_detections.append(bbox_list)
            licence_plate_texts.append(text_list)
        return licence_plate_detections, licence_plate_texts

    def detect_frame(self, frame):
        results = self.model.predict(frame)[0]
        id_name_dict = results.names
        licence_plate_list = []
        licence_plate_texts = []

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]

            if cls_name == "License_Plate":
                licence_plate_list.append(result)

                # Crop the license plate region
                x1, y1, x2, y2 = map(int, result)
                cropped_plate = frame[y1:y2, x1:x2]

                # Always apply preprocessing
                gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, None, fx=2, fy=2) # 4x resizing for better OCR accuracy
                cropped_plate = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

                # Run OCR
                # ocr_result = self.ocr.ocr(cropped_plate)
                ocr_result = self.ocr.predict(cropped_plate)
                # print(ocr_result)
                
                # Extract text safely
                # text = ocr_result[0][0][1][0] if ocr_result and ocr_result[0] else "N/A"
                if ocr_result and "rec_texts" in ocr_result[0] and ocr_result[0]["rec_texts"]:
                    text = ocr_result[0]["rec_texts"][0]
                    # print("Licence Text", text)
                else:
                    text = "N/A"
                licence_plate_texts.append(text)
        
        return licence_plate_list, licence_plate_texts

    def draw_bboxes(self, video_frames, licence_plate_detections, licence_plate_texts):
        output_video_frames = []
        for frame, plate_list, text_list in zip(video_frames, licence_plate_detections, licence_plate_texts):
            for bbox, text in zip(plate_list, text_list):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (255, 255,0), 2)
            output_video_frames.append(frame)
        return output_video_frames

