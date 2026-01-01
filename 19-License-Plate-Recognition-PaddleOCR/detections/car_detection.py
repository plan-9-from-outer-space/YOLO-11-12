# Import All the Required Libraries
import cv2
import pickle
from ultralytics import YOLO

class CarDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        car_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                car_detections = pickle.load(f)
            return car_detections
        for frame in frames:
            car_list = self.detect_frame(frame)
            car_detections.append(car_list)
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(car_detections, f)
        return car_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, iou = 0.1, conf = 0.30)[0]
        id_name_dict = results.names
        car_list = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            if cls_name == "car": # Only Detect Cars
                car_list.append(result)
        return car_list

    def draw_bboxes(self, video_frames, car_detections):
        output_video_frames = []
        for frame, car_list in zip(video_frames, car_detections):
            for idx, bbox in enumerate(car_list):
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Car", (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames
