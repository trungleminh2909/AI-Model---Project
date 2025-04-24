import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize PaddleOCR reader
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Load YOLOv8 license plate detector -> fine tuned
yolo_model = YOLO("YOLOv8_plate_full.pt")

# Detection + OCR pipeline with PaddleOCR
def detect_and_read_plate(image_path):
    # Step 1: YOLOv8 detection
    results = yolo_model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return "No license plate detected."

    # Step 2: Crop first detected plate
    x1, y1, x2, y2 = map(int, boxes[0])
    image = cv2.imread(image_path)
    plate_crop = image[y1:y2, x1:x2]
    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(plate_rgb)

    # Step 3: Smart split for two-line plates
    width, height = pil_image.size
    if height > width * 0.6:
        top_half = np.array(pil_image.crop((0, 0, width, height // 2)))
        bottom_half = np.array(pil_image.crop((0, height // 2, width, height)))

        result_top = ocr.ocr(top_half, cls=True)
        result_bot = ocr.ocr(bottom_half, cls=True)

        text_top = " ".join([line[1][0] for line in result_top[0]]) if result_top and result_top[0] else ""
        text_bot = " ".join([line[1][0] for line in result_bot[0]]) if result_bot and result_bot[0] else ""

        plate_text = f"{text_top}-{text_bot}" if text_top and text_bot else text_top + text_bot
    else:
        result = ocr.ocr(np.array(pil_image), cls=True)
        plate_text = " ".join([line[1][0] for line in result[0]]) if result and result[0] else "OCR failed"

    return plate_text

# Run inference on image
image_path = "TestImage/2.jpg"
print("Detected Plate:", detect_and_read_plate(image_path))
