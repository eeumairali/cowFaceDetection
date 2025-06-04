import os
import shutil
import csv
from ultralytics import YOLO
from typing import Dict, Any
import streamlit as st
import datetime

class YOLOTrainer:
    def __init__(self, project_dir: str, pretrained_model: str = "yolov8n.pt") -> None:
        """Initialize the YOLOTrainer with project directory and pretrained model."""
        self.project_dir = project_dir
        self.data_path = os.path.join(project_dir, "ehanced_dataset_yolov8", "data.yaml")
        self.fine_tuned_dir = os.path.join(project_dir, "models", "fine_tuned")
        self.results_dir = os.path.join(project_dir, "results")
        self.trained_dir = os.path.join(project_dir, "models", "yolov8_trained")
        os.makedirs(self.fine_tuned_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.trained_dir, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.model = YOLO(pretrained_model)

    def train_model(self, epochs: int = 50, batch: int = 16, img_size: int = 640, workers: int = 8,
                    optimizer: str = "SGD", patience: int = 10, pretrained: bool = True) -> None:
        import pandas as pd
        run_name = f"fine_tune_coco_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with st.spinner("Training YOLO model..."):
            results = self.model.train(
                data=self.data_path,
                epochs=epochs,
                batch=batch,
                imgsz=img_size,
                workers=workers,
                project=self.results_dir,
                name=run_name,
                optimizer=optimizer,
                patience=patience,
                pretrained=pretrained,
                verbose=True,
                show=True
            )
            self._move_best_model(run_name)
            # Show results_dict if available
            if hasattr(results, 'results_dict'):
                self._save_metrics(results.results_dict)
                st.subheader("Training Metrics")
                st.json(results.results_dict)
            elif hasattr(results, 'metrics') and hasattr(results.metrics, 'results_dict'):
                self._save_metrics(results.metrics.results_dict)
                st.subheader("Training Metrics")
                st.json(results.metrics.results_dict)
            else:
                st.warning("No results_dict found in YOLO training results. Metrics not saved.")
            # Show plots if available
            result_dir = os.path.join(self.results_dir, run_name)
            for plot_name in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
                plot_path = os.path.join(result_dir, plot_name)
                if os.path.exists(plot_path):
                    st.image(plot_path, caption=plot_name)
            # Show last lines of results.csv if available
            csv_path = os.path.join(result_dir, "results.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                st.subheader("Training Log (last 10 epochs)")
                st.dataframe(df.tail(10))
            # Show all images in results_dir if available
            st.subheader("Sample Images from Training Results")
            for file in os.listdir(result_dir):
                if file.endswith(('.jpg', '.png')) and not file.startswith('confusion_matrix'):
                    st.image(os.path.join(result_dir, file), caption=file, use_column_width=True)
        st.success("YOLOv8 training complete!")

    def _move_best_model(self, run_name: str) -> None:
        """Move the best model weights to the fine-tuned model directory."""
        best_weights_path = os.path.join(self.results_dir, run_name, "weights", "best.pt")
        # Save with date/time
        dt_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path_tuned = os.path.join(self.trained_dir, f"yolov8_{dt_str}.pt")
        shutil.copy(best_weights_path, model_path_tuned)
        st.info(f"Fine-tuned model saved at {model_path_tuned}")

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save training metrics as a CSV file."""
        metrics_file = os.path.join(self.results_dir, "training_metrics.csv")
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            # Handle both dict and object with .items()
            if hasattr(metrics, 'items'):
                for key, value in metrics.items():
                    writer.writerow([key, value])
            else:
                st.warning("Metrics object does not have .items(). Skipping detailed metrics save.")
        st.info(f"Training metrics saved at {metrics_file}")

    def validate_model(self) -> Dict[str, Any]:
        """Validate the fine-tuned model on the validation set."""
        with st.spinner("Validating model..."):
            val_results = self.model.val()
            self._save_metrics(val_results)
        st.success("Validation complete!")
        return val_results

    def perform_inference(self, test_image_path: str) -> Any:
        """Perform inference on a given test image."""
        with st.spinner("Running inference..."):
            results = self.model.predict(source=test_image_path, save=True)
        st.success("Inference complete!")
        return results

    @staticmethod
    def list_trained_models(trained_dir="models/yolov8_trained"):
        if not os.path.exists(trained_dir):
            return []
        return [f for f in os.listdir(trained_dir) if f.endswith('.pt')]
