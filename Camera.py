from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import base64
from fastapi.responses import RedirectResponse
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

yolo_model = YOLO("YOLOv8_plate_full.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def detect_and_read_plate_from_array(image_array: np.ndarray) -> str:
    pil_img = Image.fromarray(image_array)
    pil_img.save("temp.jpg")

    # Adjust confidence threshold if needed
    results = yolo_model("temp.jpg", conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        print("[DEBUG] No plate detected")
        return "No license plate detected."

    x1, y1, x2, y2 = map(int, boxes[0])
    plate_crop = image_array[y1:y2, x1:x2]
    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(plate_rgb)

    width, height = pil_image.size
    if height > width * 0.6:
        top_half = np.array(pil_image.crop((0, 0, width, height // 2)))
        bottom_half = np.array(pil_image.crop((0, height // 2, width, height)))
        result_top = ocr.ocr(top_half, cls=True)
        result_bot = ocr.ocr(bottom_half, cls=True)
        text_top = " ".join([line[1][0] for line in result_top[0]]) if result_top and result_top[0] else ""
        text_bot = " ".join([line[1][0] for line in result_bot[0]]) if result_bot and result_bot[0] else ""
        return f"{text_top}-{text_bot}" if text_top and text_bot else text_top + text_bot
    else:
        result = ocr.ocr(np.array(pil_image), cls=True)
        return " ".join([line[1][0] for line in result[0]]) if result and result[0] else "OCR failed"

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("qr_scan.html", encoding="utf-8").read()

class DetectPayload(BaseModel):
    image_data: str

@app.post("/detect_base64")
async def detect_base64(data: DetectPayload):
    image_bytes = base64.b64decode(data.image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    plate_text = detect_and_read_plate_from_array(image)
    return {"plate": plate_text}

class SubmitPayload(BaseModel):
    student: str
    bike: str

@app.post("/enter")
async def submit_entry(data: SubmitPayload):
    print(f"[ENTRY] Student: {data.student} - Plate: {data.bike}")
    return {"status": "entry recorded", "student": data.student, "bike": data.bike}

@app.post("/exit")
async def submit_exit(data: SubmitPayload):
    print(f"[EXIT] Student: {data.student} - Plate: {data.bike}")
    return {"status": "exit recorded", "student": data.student, "bike": data.bike}

## QR code ##
@app.get("/scan", response_class=HTMLResponse)
async def qr_page():
    return open("qr_scan.html", encoding="utf-8").read()

@app.get("/plate", response_class=HTMLResponse)
async def plate_page():
    return open("plate_detect.html", encoding="utf-8").read()

@app.get("/config")
def get_config():
    return {
        "BACKEND_IP": os.getenv("BACKEND_IP"),
        "BACKEND_PORT": os.getenv("BACKEND_PORT")
    }
