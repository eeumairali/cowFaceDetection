"""
🟦 Pipeline Flow Diagram Generator (Technical, Research-Grade)

🔹 **Purpose:**
    - Generates a high-detail, publication-ready diagram of the full cow facial recognition pipeline.
    - Includes technical details: model layers, image sizes, data splits, and identification/classification steps.
    - Suitable for research papers, presentations, and documentation.

🔹 **Key Features:**
    - Visualizes: Data input → Preprocessing → YOLOv8 Detection → Cropping → Dataset Generation → EfficientNet/ResNet Training → Identification/Prediction.
    - Shows technical details: image sizes, layer counts, model types, activation functions, data splits.
    - Output: High-res PNG for easy sharing and publication.

🔹 **Usage:**
    - Run from Streamlit or as a script to generate/update the pipeline diagram.
    - Use in research papers, presentations, and technical documentation.

"""

from graphviz import Digraph
import streamlit as st
import os

def create_pipeline_diagram(output_path="cow_detection_pipeline"):
    dot = Digraph("Cow Face Recognition Pipeline", format='png')
    dot.attr(dpi='600', rankdir='LR', size='6,6!')  # Square layout, left-to-right, fixed size

    # Section and node styles
    section_attr = {'shape': 'box', 'style': 'filled,bold', 'fontsize': '18', 'fontname': 'Arial Bold', 'fillcolor': '#263238', 'fontcolor': 'white', 'width': '2', 'height': '1'}
    node_attr = {'shape': 'box', 'style': 'filled', 'fontsize': '15', 'fontname': 'Arial', 'fontcolor': '#212121', 'width': '2', 'height': '1'}

    # 5-Layer Square Layout
    section_attr_1 = section_attr.copy(); section_attr_1['fillcolor'] = "#1976d2"
    dot.node("L1", "1. Data Loading\nCow Video/Image\n(1920x1080)", **section_attr_1)
    section_attr_2 = section_attr.copy(); section_attr_2['fillcolor'] = "#fbc02d"
    dot.node("L2", "2. Detection (YOLOv8)\n~37 layers, 640x640, SiLU", **section_attr_2)
    section_attr_3 = section_attr.copy(); section_attr_3['fillcolor'] = "#388e3c"
    dot.node("L3", "3. New Data Generation\nFace Cropping, Dataset Creation", **section_attr_3)
    section_attr_4 = section_attr.copy(); section_attr_4['fillcolor'] = "#f57c00"
    dot.node("L4", "4. Classification\nEfficientNet-B0/ResNet\n18-152 layers, 224x224", **section_attr_4)
    section_attr_5 = section_attr.copy(); section_attr_5['fillcolor'] = "#c2185b"
    dot.node("L5", "5. Output\nAnnotated Image/Video", **section_attr_5)

    # Arrange in a square (use invisible edges for layout)
    dot.edge("L1", "L2")
    dot.edge("L2", "L3")
    dot.edge("L3", "L4")
    dot.edge("L4", "L5")
    # Invisible edges to force square shape
    dot.edge("L1", "L4", style="invis")
    dot.edge("L2", "L5", style="invis")

    dot.render(output_path, format="png", cleanup=True)
    print(f"Pipeline diagram saved as {output_path}.png")

def streamlit_pipeline_diagram_page():
    st.header("Pipeline Flow Diagram (Technical, Research-Grade)")
    st.markdown("""
    <div style='background:linear-gradient(90deg,#1a237e 0%,#263238 100%);padding:20px 18px 20px 18px;border-radius:12px;'>
    <b style='color:#1976d2;font-size:1.35em;font-weight:900;'>🔎 What is this page?</b><br>
    <span style='color:#1976d2;font-size:1.15em;font-weight:700;'>Generates a high-detail, publication-ready diagram of the full cow facial recognition pipeline, including technical details for each step.</span><br>
    <b style='color:#fbc02d;font-size:1.15em;font-weight:900;'>Features:</b> <span style='color:#fbc02d;font-size:1.1em;font-weight:700;'>Visualizes data flow, model internals, image sizes, layer counts, activation functions, and data splits.</span><br>
    <b style='color:#f57c00;font-size:1.15em;font-weight:900;'>Usage:</b> <span style='color:#f57c00;font-size:1.1em;font-weight:700;'>Click 'Generate Diagram' to create/update the pipeline diagram. Useful for research papers, presentations, and documentation.</span><br>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Generate Diagram"):
        create_pipeline_diagram()
        st.success("Pipeline diagram generated!")
    if os.path.exists("cow_detection_pipeline.png"):
        st.image("cow_detection_pipeline.png", caption="Pipeline Diagram", use_container_width=True)
    st.markdown("""
    ### Technical Terms Explained
    - **YOLOv8**: A real-time object detection model. Used for cow face detection. (See: `part1_yolov8_enhanced_dataset.py`, `part2_detect_face.py`)
    - **CSPDarknet, PANet, SPPF**: Backbone and neck architectures in YOLOv8 for feature extraction and aggregation.
    - **Conv: 3x3/1x1**: Convolutional layers with 3x3 or 1x1 kernels, used for feature extraction in CNNs.
    - **SiLU (Sigmoid Linear Unit)**: Activation function used in YOLOv8. Formula: x * sigmoid(x). (See: `part1_yolov8_enhanced_dataset.py`)
    - **Train/Test Split (80/20, Stratified)**: Dataset is split into 80% for training, 20% for testing, preserving class distribution. (See: `part3_yolov8_based_dataset_generation_for_classification.py`)
    - **EfficientNet-B0**: A convolutional neural network with 18 layers, MBConv blocks, and Swish activation. Used for cow face identification. (See: `part4_resnet_model_Training.py`)
    - **MBConv**: Mobile Inverted Bottleneck Convolution, a block used in EfficientNet.
    - **Swish**: Activation function, f(x) = x * sigmoid(x), used in EfficientNet.
    - **ResNet-34/50/101**: Residual Networks with 34, 50, or 101 layers, using skip connections and ReLU activation. Used for cow face identification. (See: `part4_resnet_model_Training.py`)
    - **Skip Connections**: Connections that add the input of a layer to the output of a deeper layer, helping with gradient flow in deep networks.
    - **ReLU (Rectified Linear Unit)**: Activation function, f(x) = max(0, x), used in ResNet.
    - **CrossEntropy**: Loss function for classification tasks.
    - **Adam**: Adaptive Moment Estimation optimizer for training neural networks.
    - **Softmax**: Activation function for multi-class classification, outputs probabilities for each class.
    - **.pth, .pt**: File extensions for saved PyTorch models.
    - **Bounding Boxes, Class Labels**: Output of detection/classification, showing where and what is detected in the image/video.
    
    <br>
    <b>File Mapping:</b><br>
    - <b>YOLOv8 Training/Detection:</b> <code>part1_yolov8_enhanced_dataset.py</code>, <code>part2_detect_face.py</code><br>
    - <b>Dataset Generation:</b> <code>part3_yolov8_based_dataset_generation_for_classification.py</code><br>
    - <b>EfficientNet/ResNet Training:</b> <code>part4_resnet_model_Training.py</code><br>
    - <b>EfficientNet/ResNet Prediction:</b> <code>part5_resnet_model_predict.py</code>, <code>part6_yolo_resnet_testing_picture.py</code>, <code>part7_yolo_resnet_testing_video.py</code><br>
    """, unsafe_allow_html=True)

# Render diagram in Streamlit or as a script
if __name__ == "__main__":
    if "streamlit_run" in os.environ:
        streamlit_pipeline_diagram_page()
    else:
        create_pipeline_diagram()
