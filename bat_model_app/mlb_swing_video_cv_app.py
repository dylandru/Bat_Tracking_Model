import streamlit as st
import tempfile
import os
import base64
from ultralytics import YOLO
import cv2
import supervision as sv 
import numpy as np

# Deal with OpenMP issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def process_video(input_path, output_path):
    model = YOLO('/Bat_Tracking_Model/bat_model_weights/bat_tracking.pt') 
    
    tracker = sv.ByteTrack()
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = process_frame(model, tracker, box_annotator, label_annotator, trace_annotator, frame)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

def process_frame(model, tracker, box_annotator, label_annotator, trace_annotator, frame):
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [f"#{detections.tracker_id[i]} {results.names[detections.class_id[i]]}" for i in range(len(detections.class_id))]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    return trace_annotator.annotate(annotated_frame, detections=detections)



# Streamlit App UI

st.title("MLB Visual Bat Tracking App")

uploaded_file = st.file_uploader("Upload your MLB video", type=["mp4"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    input_video_path = tfile.name

    output_video_path = f"{tempfile.gettempdir()}/processed_{os.path.basename(input_video_path)}.mp4"

    if st.button('Process Video'):
        with st.spinner('Processing...'):
            process_video(input_video_path, output_video_path)
            st.success('Done!')

        video_file = open(output_video_path, 'rb')
        video_bytes = video_file.read()

        st.video(video_bytes)

        st.download_button(
            label="Download Processed Video",
            data=video_bytes,
            file_name="bat_tracked_video.mp4",
            mime="video/mp4"
        )
