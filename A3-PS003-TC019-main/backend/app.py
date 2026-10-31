
import base64
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "federated_global.pt"

app = FastAPI(title="Traffix Federated Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": "FedAvg YOLO26n"
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Federated model not found in backend/models/"
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image.")

    raw = await file.read()

    if not raw or len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Invalid image or file too large.")

    image = cv2.imdecode(
        np.frombuffer(raw, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image.")

    start = time.perf_counter()

    result = model.predict(
        image,
        imgsz=416,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    inference_ms = (time.perf_counter() - start) * 1000

    annotated = result.plot()
    success, encoded = cv2.imencode(".jpg", annotated)

    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    counts = Counter()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = result.names[class_id]
            confidence = float(box.conf.item())

            counts[label] += 1

            detections.append({
                "class": label,
                "confidence": round(confidence, 4),
                "box_xyxy": [
                    round(float(v), 1)
                    for v in box.xyxy[0].tolist()
                ]
            })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "class_counts": dict(counts),
        "detections": detections,
        "inference_ms": round(inference_ms, 2),
        "annotated_image": base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
    }
