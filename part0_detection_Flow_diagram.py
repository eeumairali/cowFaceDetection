"""
🟦 Pipeline Flow Diagram Generator

🔹 **Purpose:**
    - Generates a visual diagram of the full cow facial recognition pipeline.
    - Helps users and supervisors understand the step-by-step process.

🔹 **Key Features:**
    - Visualizes data flow: video/image input → YOLOv8 detection → dataset generation → EfficientNet/ResNet training → prediction.
    - Can be extended to show more details (e.g., model internals, data splits).
    - Output is a PNG image for easy sharing and presentation.

🔹 **Usage:**
    - Run from the Streamlit app or as a script to generate/update the pipeline diagram.
    - Useful for presentations and documentation.

"""

from graphviz import Digraph

def create_pipeline_diagram(output_path="cow_detection_pipeline"): 
    dot = Digraph("Cow Face Detection Pipeline", format='png')
    dot.attr(dpi='600')  # High-resolution output

    # Define common attributes for square, colorful boxes
    node_attr = {'shape': 'box', 'style': 'filled', 'fontsize': '16', 'fontname': 'Arial Bold'}
    section_attr = {'shape': 'box', 'style': 'filled,bold', 'fontsize': '18', 'fontname': 'Arial Bold', 'fillcolor': 'gray'}

    colors = ["lightblue", "lightgreen", "orange", "pink", "yellow", "cyan"]

    # Data Preparation
    dot.node("P1", "Data Preparation", **section_attr)
    dot.node("A", "Cow Video", fillcolor=colors[0], **node_attr)
    dot.node("B", "Extract Frames", fillcolor=colors[1], **node_attr)
    dot.node("C", "Merge with Public Dataset", fillcolor=colors[2], **node_attr)
    dot.node("F", "Preprocessing", fillcolor=colors[3], **node_attr)
    dot.node("G", "Train-Test Split", fillcolor=colors[4], **node_attr)

    # Model Training
    dot.node("P2", "Model Training", **section_attr)
    dot.node("I", "Load Pretrained Model", fillcolor=colors[0], **node_attr)
    dot.node("J", "Fine-Tuning", fillcolor=colors[1], **node_attr)
    dot.node("K", "Compute Loss & Backpropagation", fillcolor=colors[2], **node_attr)
    dot.node("KK", "Optimization", fillcolor=colors[3], **node_attr)
    dot.node("L", "Model Evaluation", fillcolor=colors[4], **node_attr)
    dot.node("M", "Save Best Model", fillcolor=colors[5], **node_attr)

    # Model Testing
    dot.node("P3", "Model Testing", **section_attr)
    dot.node("N", "Load Model", fillcolor=colors[0], **node_attr)
    dot.node("O", "YOLO Detection", fillcolor=colors[1], **node_attr)
    dot.node("P", "Save Predictions", fillcolor=colors[2], **node_attr)

    # Define edges
    dot.edge("P1", "A")
    dot.edge("A", "B")
    dot.edge("B", "C")
    dot.edge("C", "F")
    dot.edge("F", "G")

    # Training connections
    dot.edge("P2", "I")
    dot.edge("I", "J")
    dot.edge("J", "K")
    dot.edge("K", "KK")
    dot.edge("KK", "L")
    dot.edge("L", "M")

    # Testing connections
    dot.edge("P3", "N")
    dot.edge("N", "O")
    dot.edge("O", "P")

    # Render diagram
    dot.render(output_path, format="png", cleanup=True)
    print(f"Pipeline diagram saved as {output_path}.png")

if __name__ == "__main__":
    create_pipeline_diagram()
