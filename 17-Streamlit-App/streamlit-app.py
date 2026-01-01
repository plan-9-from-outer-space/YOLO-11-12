# Import All the Required Libraries
import cv2
import streamlit as st
import sys
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory of the current file
# ROOT = FILE.parent
ROOT = FILE.parent.parent
# ROOT = C:\Courses\2025\Udemy\YOLO-11-12
# CWD = C:\Courses\2025\Udemy\YOLO-11-12\17-Streamlit-App

# Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get the relative path of the root directory with respect to the current working directory
# ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'

SOURCES_LIST = [IMAGE, VIDEO]

# Image Config
IMAGES_DIR = ROOT/'images'
DEFAULT_IMAGE = IMAGES_DIR/'image1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR/'detectedimage1.jpg'

# Videos Config
VIDEO_DIR = ROOT/'videos'
VIDEOS_DICT = {
    'street repair': VIDEO_DIR/'video-street-repair.mp4',
    'bicycle ride': VIDEO_DIR/'video-bicycle-ride.mp4'
}

# Model Configurations
MODEL_DIR = ROOT/'models'
DETECTION_MODEL = MODEL_DIR/'yolo11n.pt'
# DETECTION_MODEL = MODEL_DIR/'custom_model_weight.pt' # Custom model weights
SEGMENTATION_MODEL = MODEL_DIR/'yolo11n-seg.pt'
POSE_ESTIMATION_MODEL = MODEL_DIR/'yolo11n-pose.pt'

#################################
# STREAMLIT Configuration
#################################

# Page Layout
st.set_page_config(
    page_title = "YOLO11",
    page_icon = "🤖"
)

# Header
st.header("Object Detection using YOLO11")

# SideBar
st.sidebar.header("Model Configurations")

# Choose Model: Detection, Segmentation or Pose Estimation
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])

# Select Confidence Value
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40)) / 100

# Selecting Detection, Segmentation, Pose Estimation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
elif model_type ==  'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)

# Load the YOLO Model
# print(model_path)
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the sepcified model path: {model_path}")
    st.error(e)

# Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    label = "Select Source", 
    options = SOURCES_LIST
)

source_image = None

if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader(
        label = "Choose an Image....", 
        type = ("jpg", "png", "jpeg", "bmp", "webp")
    )
    col1, col2 = st.columns(2)

# Streamlit note:
# 
# The width of the image element. This can be one of the following:
#     "content" (default): The width of the element matches the width of its content, but doesn't exceed the width of the parent container.
#     "stretch": The width of the element matches the width of the parent container.
#     An integer specifying the width in pixels: The element has a fixed width. If the specified width is greater 
#       than the width of the parent container, the width of the element matches the width of the parent container.
# When using an SVG image without a default width, use "stretch" or an integer.

    with col1:
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(image = default_image_path, caption = "Default Image", width="stretch") # use_column_width = True)
            else:
                uploaded_image = Image.open(source_image)
                st.image(image = source_image, caption = "Uploaded Image", width="stretch") # use_column_width = True)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)
    
    with col2:
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption="Detected Image", width="stretch") # use_column_width = True)
            else:
                if st.sidebar.button("Detect Objects"):
                    result = model.predict(uploaded_image, conf=confidence_value)
                    boxes = result[0].boxes
                    result_plotted = result[0].plot()[:, :, ::-1]
                    st.image(result_plotted, caption="Detected Image", width="stretch") # use_column_width = True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
        except Exception as e:
            st.error("Error Occured While Opening the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox(
        label = "Choose a Video...", 
        options = VIDEOS_DICT.keys()
    )
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if st.sidebar.button("Detect Video Objects"):
            # pass
            try:
                video_cap = cv2.VideoCapture(
                    filename = str(VIDEOS_DICT.get(source_video)))
                st_frame = st.empty()
                while (video_cap.isOpened()):
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        # Predict the objects in the image using YOLO11
                        result = model.predict(source=image, conf=confidence_value)
                        # Plot the detected objects on the video frame
                        result_plotted = result[0].plot()
                        st_frame.image(
                            image = result_plotted, 
                            caption = "Detected Video",
                            channels = "BGR",
                            width = "stretch") # use_column_width=True
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Video" + str(e))
