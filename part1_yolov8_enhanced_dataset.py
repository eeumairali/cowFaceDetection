import os
import shutil
import csv
from ultralytics import YOLO
from typing import Dict, Any

class YOLOTrainer:
    def __init__(self, project_dir: str, pretrained_model: str = "yolov8n.pt") -> None:
        """Initialize the YOLOTrainer with project directory and pretrained model."""
        self.project_dir = project_dir
        self.data_path = os.path.join(project_dir, "ehanced_dataset_yolov8", "data.yaml")
        self.fine_tuned_dir = os.path.join(project_dir, "models", "fine_tuned")
        self.results_dir = os.path.join(project_dir, "results")
        os.makedirs(self.fine_tuned_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.pretrained_model = pretrained_model
        self.model = YOLO(pretrained_model)

    def train_model(self, epochs: int = 50, batch: int = 16, img_size: int = 640, workers: int = 8,
                    optimizer: str = "SGD", patience: int = 10, pretrained: bool = True) -> None:
        """Train the YOLO model on the custom dataset."""
        results = self.model.train(
            data=self.data_path,
            epochs=epochs,
            batch=batch,
            imgsz=img_size,
            workers=workers,
            project=self.results_dir,
            name="fine_tune_coco",
            optimizer=optimizer,
            patience=patience,
            pretrained=pretrained
        )
        self._move_best_model()
        self._save_metrics(results.metrics)

    def _move_best_model(self) -> None:
        """Move the best model weights to the fine-tuned model directory."""
        best_weights_path = os.path.join(self.results_dir, "fine_tune_coco", "weights", "best.pt")
        model_path_tuned = os.path.join(self.fine_tuned_dir, "yolov8_fine_tuned_cow_face.pt")
        shutil.copy(best_weights_path, model_path_tuned)
        print(f"Fine-tuned model saved at {model_path_tuned}")

    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save training metrics as a CSV file."""
        metrics_file = os.path.join(self.results_dir, "training_metrics.csv")
        with open(metrics_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        print(f"Training metrics saved at {metrics_file}")

    def validate_model(self) -> Dict[str, Any]:
        """Validate the fine-tuned model on the validation set."""
        val_results = self.model.val()
        self._save_metrics(val_results)
        return val_results

    def perform_inference(self, test_image_path: str) -> Any:
        """Perform inference on a given test image."""
        return self.model.predict(source=test_image_path, save=True)

# Usage example:
if __name__ == "__main__":
    project_directory = "/mnt/c/Users/eeuma/Desktop/cow_facial_recognition_yolo_imagenet"
    trainer = YOLOTrainer(project_directory)
    trainer.train_model()
    trainer.validate_model()
    test_image = os.path.join(project_directory, "frame_0000_jpg.rf.149f754f06c3550a5b7617649e082e3c.jpg")
    trainer.perform_inference(test_image)
    print("Inference complete. Check the output folder for predictions.")
