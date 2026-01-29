import cv2
import json
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO("MJ_tile_detection_yolo11-obb.pt")

def obb_predict(input_img: np.ndarray):
    """
    Gradio will pass an RGB numpy array (H, W, 3).
    Return the annotated RGB array.
    """
    if input_img is None:
        return None

    # Convert RGB (Gradio) -> BGR (OpenCV/YOLO)
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Inference
    results = model(img_bgr, imgsz=1920, conf=0.5, verbose=False)

    # Process first image (single-image interface)
    r = results[0]
    annotated_img = r.plot()

    # Convert BGR -> RGB for Gradio output
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    return annotated_img_rgb, results[0]

if __name__ == "__main__":
    test_dir = Path("test")
    for img_path in sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.JPG")):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, result = obb_predict(img_rgb)
        output_path = Path("output") / img_path.stem
        output_path.parent.mkdir(exist_ok=True)
        with open(f"{output_path}.json", "w") as f:
            json.dump(json.loads(result.to_json()), f, indent=2)
        print(f"Saved: {output_path}.json")