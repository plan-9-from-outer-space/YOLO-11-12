
from utils import (read_video, save_video)
from detections import CarDetection, LicencePlateDetection

def main():
    # Input Video Path
    input_video_path = "input_videos/video4.mp4"

    # Read Video frames
    video_frames = read_video(input_video_path)

    # Detect Cars
    car_detector = CarDetection(model_path="models/yolo11n.pt")
    car_detections = car_detector.detect_frames(
        frames=video_frames, read_from_stub=True, stub_path="tracker_stubs/car_detection.pkl")

    # Detect Licence Plates
    # Used save model from fine-tuning, as pre-trained yolo model does not detect license plates
    licence_plate_detector = LicencePlateDetection(model_path="models/best.pt")
    licence_plate_detections, licence_plate_texts = licence_plate_detector.detect_frames(video_frames)

    # Draw Car Bounding Boxes
    output_video_frames = car_detector.draw_bboxes(video_frames, car_detections)

    # Draw Licence Plate Bounding Boxes
    output_video_frames = licence_plate_detector.draw_bboxes(
        output_video_frames, licence_plate_detections, licence_plate_texts)
    
    # Save the Output Video
    save_video(output_video_frames, output_video_path="output_videos/output_video3.avi")

if __name__ == "__main__":
    main()
