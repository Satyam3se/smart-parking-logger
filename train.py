"""
Smart Parking Logger -- YOLOv8 License Plate Model Training & Fine-Tuning Pipeline
================================================================================
Trains or fine-tunes a YOLOv8 object detection model on license plate datasets
and outputs the optimized weights directly into the project's models/ folder.

Usage:
    python train.py [--epochs 25] [--imgsz 640] [--batch 8] [--weights yolov8n.pt]
    python train.py --quick-test   # 1 epoch fast sanity run
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = BASE_DIR / "data" / "data.yaml"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_OUTPUT_WEIGHTS = DEFAULT_MODELS_DIR / "license_plate_detector.pt"
RUNS_DIR = BASE_DIR / "runs" / "train"

def train_model(
    data_yaml: str = str(DEFAULT_DATA_YAML),
    base_weights: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch: int = 8,
    device: str = None,
    output_weights: str = str(DEFAULT_OUTPUT_WEIGHTS),
    project_dir: str = str(RUNS_DIR),
    run_name: str = "smart_parking_plate_model",
    quick_test: bool = False
):
    """
    Executes YOLOv8 training/fine-tuning loop on license plate dataset.
    """
    DEFAULT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-detect device if not specified
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("🚀 SMART PARKING LOGGER - MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"• Dataset Config  : {data_yaml}")
    print(f"• Base Weights    : {base_weights}")
    print(f"• Target Output   : {output_weights}")
    print(f"• Device          : {device} ({'GPU Acceleration' if device != 'cpu' else 'CPU Mode'})")
    print(f"• Image Size      : {imgsz}")
    print(f"• Batch Size      : {batch}")
    print(f"• Epochs          : {1 if quick_test else epochs}")
    print("=" * 65)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset config not found at: {data_yaml}")

    # Check base weights: if file doesn't exist locally, YOLO will download base yolov8n.pt
    model = YOLO(base_weights)

    actual_epochs = 1 if quick_test else epochs

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=actual_epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        verbose=True,
        workers=2 if device == "cpu" else 4,
        val=True
    )

    # Locate best.pt
    best_weights_path = Path(project_dir) / run_name / "weights" / "best.pt"
    if not best_weights_path.exists():
        # Fallback to last.pt if best.pt is absent (e.g. 1 epoch without validation improvement)
        best_weights_path = Path(project_dir) / run_name / "weights" / "last.pt"

    if best_weights_path.exists():
        shutil.copy2(best_weights_path, output_weights)
        print("\n" + "=" * 65)
        print(f"✅ Training completed successfully!")
        print(f"📦 Best weights saved to: {output_weights}")
        print("=" * 65)
        return str(output_weights)
    else:
        print(f"⚠️ Warning: Could not find trained weights at {best_weights_path}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on License Plate Dataset for Smart Parking Logger")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights: yolov8n.pt or models/license_plate_detector.pt")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (pixels)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', '0', etc.)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_WEIGHTS), help="Destination path for best.pt")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 epoch test run to verify pipeline")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        data_yaml=args.data,
        base_weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_weights=args.output,
        quick_test=args.quick_test
    )
