import streamlit as st
import os
from PIL import Image

# Import classes from your pipeline scripts
from part0_detection_Flow_diagram import create_pipeline_diagram
from part1_yolov8_enhanced_dataset import YOLOTrainer
from part2_detect_face import YOLOProcessor
from part3_yolov8_based_dataset_generation_for_classification import CowFaceDatasetPreparer
from part4_resnet_model_Training import CowFacialRecognition as ResnetTrainer
from part5_resnet_model_predict import CowFacialRecognition as ResnetPredictor
from part6_yolo_resnet_testing_picture import CowFacialRecognition as YoloResnetPic
from part7_yolo_resnet_testing_video import CowFaceRecognizer as YoloResnetVid

st.set_page_config(page_title="Cow Facial Recognition Pipeline", layout="wide")
st.title("🐮 Cow Facial Recognition Pipeline")

TABS = [
    "0. Pipeline Diagram",
    "1. YOLOv8 Enhanced Dataset Training",
    "2. Detect Face (YOLOv8)",
    "3. Dataset Generation for Classification",
    "4. ResNet Model Training",
    "5. ResNet Model Predict",
    "6. YOLO+ResNet Test (Picture)",
    "7. YOLO+ResNet Test (Video)"
]
tab = st.sidebar.radio("Select Pipeline Step", TABS)

if tab == TABS[0]:
    st.header("Pipeline Flow Diagram")
    if st.button("Generate Diagram"):
        create_pipeline_diagram()
        st.success("Pipeline diagram generated!")
    if os.path.exists("cow_detection_pipeline.png"):
        st.image("cow_detection_pipeline.png", caption="Pipeline Diagram", use_container_width=True)

elif tab == TABS[1]:
    st.header("YOLOv8 Enhanced Dataset Training")
    project_dir = st.text_input("Project Directory", value=os.getcwd())
    epochs = st.number_input("Epochs", 1, 200, 50)
    if st.button("Train YOLOv8 Model"):
        trainer = YOLOTrainer(project_dir)
        trainer.train_model(epochs=epochs)
        st.success("YOLOv8 training complete!")

elif tab == TABS[2]:
    st.header("Detect Face with YOLOv8")
    yolo_trained_dir = "models/yolov8_trained"
    # make a dropdown to select a trained YOLOv8 model from the trained directory
    st.write("Select a trained YOLOv8 model:")
    available_models = YOLOTrainer.list_trained_models(yolo_trained_dir)
    model_path = st.selectbox("Trained YOLOv8 Model", available_models, key="yolo_model_select")
    if model_path:
        model_path = os.path.join(yolo_trained_dir, model_path)
    else:
        st.error("Please select a YOLOv8 model.")
        model_path = None
    processor = YOLOProcessor(model_path)
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png"])
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_img and st.button("Detect Face in Image"):
        img_path = f"detected_face_yolov8/part2_demo.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())
        processor.process_image(img_path)
        st.image("detected_face_yolov8/part2_demo_out.png", caption="Detected Face")
    if uploaded_vid and st.button("Detect Face in Video"):
        vid_path = f"detected_face_yolov8/part2_demo_video.mp4"
        with open(vid_path, "wb") as f:
            f.write(uploaded_vid.read())
        processor.process_video(vid_path)
        st.video("detected_face_yolov8/part2_demo_out.mp4")

elif tab == TABS[3]:
    st.header("Dataset Generation for Classification")
    video_folder = st.text_input("Video Folder", value="cow_faces_videos")
    output_folder = st.text_input("Output Folder", value="dataset_for_classification")
    yolo_trained_dir = "models/yolov8_trained"
    st.write("Select a trained YOLOv8 model:")
    available_models = YOLOTrainer.list_trained_models(yolo_trained_dir)
    yolo_model_path = st.selectbox("Trained YOLOv8 Model", available_models, key="yolo_model_select_dataset")
    if yolo_model_path:
        yolo_model_path = os.path.join(yolo_trained_dir, yolo_model_path)
    else:
        st.error("Please select a YOLOv8 model.")
        yolo_model_path = None
    if st.button("Prepare Dataset"):
        preparer = CowFaceDatasetPreparer(video_folder, output_folder, yolo_model_path)
        preparer.prepare_dataset()
        st.success(f"Dataset prepared in {output_folder}!")

elif tab == TABS[4]:
    st.header("ResNet Model Training")
    dataset_path = st.text_input("Dataset Path", value="dataset_for_classification")
    output_model_dir = st.text_input("Output Model Directory", value="models/efficientnet_trained")
    if st.button("Train ResNet Model"):
        trainer = ResnetTrainer(dataset_path, trained_dir=output_model_dir)
        trainer.train()
        st.success(f"ResNet model trained and saved in {output_model_dir}!")

elif tab == TABS[5]:
    st.header("ResNet Model Predict")
    train_dir = st.text_input("Train Dir", value="dataset_for_classification/train")
    efficientnet_trained_dir = st.text_input("EfficientNet Model Directory", value="models/efficientnet_trained")
    available_models = ResnetTrainer.list_trained_models(efficientnet_trained_dir)
    model_path = st.selectbox("Select EfficientNet Model", available_models, key="efficientnet_model_select_predict")
    if model_path:
        model_path = os.path.join(efficientnet_trained_dir, model_path)
    else:
        st.error("Please select an EfficientNet model.")
        model_path = None
    uploaded_img = st.file_uploader("Upload Image for Prediction", type=["jpg", "png"], key="resnet_predict")
    output_img_path = st.text_input("Output Image Path", value="results/resnet_predict_out.png")
    if uploaded_img and st.button("Predict Class"):
        img_path = "temp_predict_img.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())
        predictor = ResnetPredictor(train_dir, model_path)
        pred_class = predictor.predict(img_path)
        st.success(f"Predicted Class: {pred_class}")
        if os.path.exists(output_img_path):
            st.image(output_img_path, caption="Prediction Output")

elif tab == TABS[6]:
    st.header("YOLO+ResNet Test (Picture)")
    train_dir = st.text_input("Train Dir", value="dataset_for_classification/train", key="pic_train")
    efficientnet_trained_dir = st.text_input("EfficientNet Model Directory", value="models/efficientnet_trained", key="pic_effnet_dir")
    yolo_trained_dir = st.text_input("YOLOv8 Model Directory", value="models/yolov8_trained", key="pic_yolo_dir")
    available_effnet = YoloResnetPic.list_trained_models(efficientnet_trained_dir)
    available_yolo = YOLOTrainer.list_trained_models(yolo_trained_dir)
    model_path = st.selectbox("Select EfficientNet Model", available_effnet, key="efficientnet_model_select_pic")
    yolo_model_path = st.selectbox("Select YOLOv8 Model", available_yolo, key="yolo_model_select_pic")
    if model_path:
        model_path = os.path.join(efficientnet_trained_dir, model_path)
    else:
        st.error("Please select an EfficientNet model.")
        model_path = None
    if yolo_model_path:
        yolo_model_path = os.path.join(yolo_trained_dir, yolo_model_path)
    else:
        st.error("Please select a YOLOv8 model.")
        yolo_model_path = None
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "png"], key="pic_test")
    output_img_path = st.text_input("Output Image Path", value="results/yolores.png", key="pic_out_path")
    if uploaded_img and st.button("Run YOLO+ResNet on Image"):
        img_path = "temp_pic_test.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.read())
        recognizer = YoloResnetPic(train_dir, model_path, yolo_model_path)
        recognizer.predict(img_path, output_img_path)
        if os.path.exists(output_img_path):
            st.image(output_img_path, caption="YOLO+ResNet Output")

elif tab == TABS[7]:
    st.header("YOLO+ResNet Test (Video)")
    train_dir = st.text_input("Train Dir", value="dataset_for_classification/train", key="vid_train")
    efficientnet_trained_dir = st.text_input("EfficientNet Model Directory", value="models/efficientnet_trained", key="vid_effnet_dir")
    yolo_trained_dir = st.text_input("YOLOv8 Model Directory", value="models/yolov8_trained", key="vid_yolo_dir")
    available_effnet = YoloResnetVid.list_trained_models(efficientnet_trained_dir)
    available_yolo = YOLOTrainer.list_trained_models(yolo_trained_dir)
    model_path = st.selectbox("Select EfficientNet Model", available_effnet, key="efficientnet_model_select_vid")
    yolo_model_path = st.selectbox("Select YOLOv8 Model", available_yolo, key="yolo_model_select_vid")
    if model_path:
        model_path = os.path.join(efficientnet_trained_dir, model_path)
    else:
        st.error("Please select an EfficientNet model.")
        model_path = None
    if yolo_model_path:
        yolo_model_path = os.path.join(yolo_trained_dir, yolo_model_path)
    else:
        st.error("Please select a YOLOv8 model.")
        yolo_model_path = None
    uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi"], key="vid_test")
    output_vid_path = st.text_input("Output Video Path", value="results/labeled_output10.mp4", key="vid_out_path")
    if uploaded_vid and st.button("Run YOLO+ResNet on Video"):
        vid_path = "temp_vid_test.mp4"
        with open(vid_path, "wb") as f:
            f.write(uploaded_vid.read())
        recognizer = YoloResnetVid(train_dir, model_path, yolo_model_path, vid_path, output_vid_path)
        recognizer.process_video()
        if os.path.exists(output_vid_path):
            st.video(output_vid_path)
