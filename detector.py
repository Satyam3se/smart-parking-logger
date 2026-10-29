import os
import re
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "license_plate_detector.pt")
FALLBACK_TRACKX_PATH = os.path.join(os.path.dirname(BASE_DIR), "track-x3", "pretrained_weights", "license_plate_detector.pt")

def list_available_models() -> List[str]:
    """Scan and return all available .pt model weight paths in models/ and system."""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".pt"):
                models.append(os.path.join(MODELS_DIR, f))
    if not models and os.path.exists(FALLBACK_TRACKX_PATH):
        models.append(FALLBACK_TRACKX_PATH)
    return models

class PlateDetectorOCR:
    def __init__(self, model_path: Optional[str] = None):
        if not model_path or not os.path.exists(model_path):
            if os.path.exists(DEFAULT_MODEL_PATH):
                self.model_path = DEFAULT_MODEL_PATH
            elif os.path.exists(FALLBACK_TRACKX_PATH):
                self.model_path = FALLBACK_TRACKX_PATH
            else:
                self.model_path = "yolov8n.pt"
        else:
            self.model_path = model_path
        self.model = None
        self.reader = None
        self._load_yolo()
        self._load_ocr()

    def _load_yolo(self):
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            raise FileNotFoundError(f"YOLO Plate Detector weights not found at: {self.model_path}")

    def _load_ocr(self):
        import easyocr
        # Initialize easyocr reader with GPU fallback to CPU
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Enhance plate crop image for robust OCR extraction:
        - Resize/upscale for clear character contours
        - Grayscale conversion
        - Bilateral filter / Gaussian blur for noise removal
        - Contrast adjustment (CLAHE) and Otsu thresholding
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        h, w = plate_crop.shape[:2]
        # Upscale if crop is too small
        target_h = 100
        if h < target_h:
            scale = target_h / float(h)
            plate_crop = cv2.resize(plate_crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        
        # Adaptive histogram equalization (CLAHE) for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_boost = clahe.apply(gray)

        # Bilateral filter to smooth noise while preserving character edges
        denoised = cv2.bilateralFilter(contrast_boost, 11, 17, 17)

        # Otsu thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Cleans OCR text to extract clean alphanumeric license plate strings.
        Removes special symbols and spaces, keeping standard A-Z, 0-9.
        """
        if not raw_text:
            return ""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        return cleaned

    def detect_and_recognize(self, frame: np.ndarray, conf_thresh: float = 0.25) -> List[Dict[str, Any]]:
        """
        Runs YOLOv8 detector on the frame, crops plate regions,
        performs preprocessing and EasyOCR text extraction.
        
        Returns a list of dicts:
        [
            {
                "bbox": [x1, y1, x2, y2],
                "conf": float,
                "plate_crop": np.ndarray (BGR),
                "preprocessed_crop": np.ndarray (Binary/Gray),
                "raw_text": str,
                "plate_text": str,
                "ocr_confidence": float
            }, ...
        ]
        """
        if self.model is None or frame is None:
            return []

        results = self.model.predict(source=frame, conf=conf_thresh, verbose=False)
        detections = []

        h_frame, w_frame = frame.shape[:2]

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())

                x1, y1, x2, y2 = xyxy
                # Clip bounds to frame dimensions
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue

                plate_crop = frame[y1:y2, x1:x2].copy()
                preprocessed = self.preprocess_plate(plate_crop)

                # EasyOCR inference on preprocessed crop
                ocr_results = self.reader.readtext(preprocessed)
                
                # Combine detected text segments sorted by position
                raw_text_parts = []
                conf_sum = 0.0
                valid_parts = 0

                for item in ocr_results:
                    _, text, score = item
                    raw_text_parts.append(text)
                    conf_sum += score
                    valid_parts += 1

                raw_text = " ".join(raw_text_parts)
                cleaned_text = self.clean_plate_text(raw_text)

                # Fallback to direct colored plate crop if preprocessed yielded nothing
                if not cleaned_text:
                    direct_ocr = self.reader.readtext(plate_crop)
                    raw_text_parts_direct = []
                    for item in direct_ocr:
                        _, text, score = item
                        raw_text_parts_direct.append(text)
                        conf_sum += score
                        valid_parts += 1
                    raw_text = " ".join(raw_text_parts_direct)
                    cleaned_text = self.clean_plate_text(raw_text)

                avg_conf = (conf_sum / valid_parts) if valid_parts > 0 else 0.0

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "detector_conf": round(conf, 3),
                    "plate_crop": plate_crop,
                    "preprocessed_crop": preprocessed,
                    "raw_text": raw_text,
                    "plate_text": cleaned_text,
                    "ocr_confidence": round(avg_conf, 3)
                })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Renders bounding boxes and predicted plate text on the input frame.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["plate_text"] if det["plate_text"] else f"Plate: {det['detector_conf']:.2f}"
            
            # Green border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 118), 3)

            # Label badge
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (0, 230, 118), -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        return annotated
