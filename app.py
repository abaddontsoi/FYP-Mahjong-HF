import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import gradio as gr

# --------------------------- CONFIG ---------------------------
DETECTOR_PATH   = "MJ_tile_detection_yolo11.pt"                # detection .pt
CLASSIFIER_PATH = "MJ_tile_detection_classification_yolo11.pt" # classification .pt
SAVE_DIR        = Path("pipeline_output")
SAVE_DIR.mkdir(exist_ok=True)

DET_CONF = 0.5   # detection confidence
CLS_CONF = 0.5   # classification confidence

# --------------------------- MODELS (load once) ---------------
detector   = YOLO(DETECTOR_PATH)      # task='detect'
detector.zero_grad()
classifier = YOLO(CLASSIFIER_PATH)    # task='classify'

print("Classifier classes:", classifier.names)


def pipeline(input_img: np.ndarray):
    """
    Gradio will pass an RGB numpy array (H, W, 3).
    Return the annotated RGB array.
    """
    if input_img is None:
        return None

    # Convert RGB (Gradio) -> BGR (OpenCV/YOLO)
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # Resize to 1920 x 1080
    # img_bgr = cv2.resize(img_bgr, (1920, 1080), interpolation=cv2.INTER_AREA)

    # --------------------------- PIPELINE ------------------------
    results = detector(img_bgr, imgsz=1920, conf=DET_CONF, verbose=False)  # list[Results]

    # Process first image (single-image interface)
    r = results[0]
    img = r.orig_img.copy()  # BGR
    all_cls_res = []
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_conf = box.conf.item()

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Classify crop
        cls_res = classifier(crop, conf=CLS_CONF, verbose=False)[0]

        if cls_res.probs is None:
            tile_type = "unknown"
            cls_conf = 0.0
        else:
            tile_type = cls_res.names[cls_res.probs.top1]
            cls_conf = cls_res.probs.top1conf.item()
            all_cls_res.append(cls_res)

        # Draw on original image
        color = (0, 255, 0) if tile_type != "unknown" else (0, 0, 255)
        label = f"{tile_type} {cls_conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            2,
        )

    # Optionally save annotated image
    # annotated_path = SAVE_DIR / "last_annotated.jpg"
    # cv2.imwrite(str(annotated_path), img)

    # Convert BGR back to RGB for Gradio
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, [cls_res.names[cls_res.probs.top1] for cls_res in all_cls_res]


# --------------------------- GRADIO APP ------------------------
demo = gr.Interface(
    fn=pipeline,
    inputs=gr.Image(type="numpy", label="Upload Mahjong Image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Detection + Classification"),
        gr.TextArea(label="Detected tiles"),
        ],
    title="Mahjong Tile Detection + Classification",
    description="Upload an image to detect Mahjong tiles and classify each tile.",
)


demo.launch()