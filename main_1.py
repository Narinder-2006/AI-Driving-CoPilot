"""
AI Driving Co-Pilot — Backend (Redesigned)
==========================================
Key architectural change: NO frame-by-frame inference.
- Webcam/video plays natively in the browser (no server involvement).
- The browser snapshots ONE frame and POSTs it as JPEG when the user
  clicks "Analyse Scene" or sends a chat query.
- The server runs YOLO + UNet + BiLSTM + Qwen on that single frame
  and returns JSON.  No streaming, no polling loop.
"""

import os
import io
import base64
import logging

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pipeline import (
    CONFIG,
    load_all_models,
    run_yolo,
    run_unet_array,
    run_bilstm,
    run_transformer,
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(title="AI Driving Co-Pilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Model store  (populated once at startup)
# ─────────────────────────────────────────────
MODELS: dict = {
    "loaded":     False,
    "yolo":       None,
    "unet":       None,
    "bilstm":     None,
    "vectorizer": None,
    "tokenizer":  None,
    "phi3":       None,
}


# ─────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Loading models …")
    try:
        yolo, unet, bilstm, vectorizer, tokenizer, phi3 = load_all_models(CONFIG)
        MODELS.update(
            yolo=yolo, unet=unet, bilstm=bilstm,
            vectorizer=vectorizer, tokenizer=tokenizer,
            phi3=phi3, loaded=True,
        )
        logger.info("✅ All models ready!")
    except Exception as exc:
        logger.error(f"❌ Model loading failed: {exc}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    if MODELS["phi3"] is not None:
        del MODELS["phi3"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# Helper — decode uploaded JPEG bytes → BGR numpy
# ─────────────────────────────────────────────
def bytes_to_bgr(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame


def bgr_to_b64(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not encode frame to JPEG")
    return base64.b64encode(buf).decode()


# ─────────────────────────────────────────────
# Core inference — called once per user action
# ─────────────────────────────────────────────
def run_inference_on_frame(frame_bgr: np.ndarray, user_query: str, imgsz: int = 320):
    """
    Run the full pipeline on a single BGR frame.
    Returns a plain dict ready to be sent as JSON.
    """
    # ── 1. YOLO ──────────────────────────────
    results = MODELS["yolo"](frame_bgr, imgsz=imgsz, verbose=False,
                              half=torch.cuda.is_available())
    result = results[0]
    yolo_annotated = result.plot()          # annotated BGR image

    detections: list[str] = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = MODELS["yolo"].names[cls_id]
            detections.append(f"{label} ({conf:.0%})")
    detection_str = ", ".join(detections) if detections else "no objects detected"

    # ── 2. UNet ──────────────────────────────
    unet_overlay, _mask, drivable_pct = run_unet_array(
        MODELS["unet"], frame_bgr, CONFIG["unet_input_size"]
    )

    # ── 3. Blend annotations ─────────────────
    annotated = cv2.addWeighted(unet_overlay, 0.55, yolo_annotated, 0.45, 0)

    # ── 4. BiLSTM intent ─────────────────────
    _idx, bilstm_label, bilstm_emoji, bilstm_conf, _probs = run_bilstm(
        MODELS["bilstm"], MODELS["vectorizer"],
        user_query, CONFIG["bilstm_num_classes"]
    )

    # ── 5. Qwen reasoning ────────────────────
    decision, vision_context = run_transformer(
        MODELS["tokenizer"], MODELS["phi3"],
        user_query, detections, drivable_pct, bilstm_label
    )

    return {
        "annotated_frame": bgr_to_b64(annotated),
        "detections":      detection_str,
        "drivable_pct":    drivable_pct,
        "bilstm_label":    bilstm_label,
        "bilstm_emoji":    bilstm_emoji,
        "bilstm_conf":     f"{bilstm_conf:.0%}",
        "decision":        decision,
        "vision_context":  vision_context,
    }


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/api/status")
async def status():
    """Health check — is the server ready?"""
    return {
        "status":        "ready" if MODELS["loaded"] else "loading",
        "models_loaded": MODELS["loaded"],
    }


@app.post("/api/analyse")
async def analyse(
    frame: UploadFile = File(..., description="JPEG snapshot from the browser"),
    query: str = "Is it safe to drive?",
    imgsz: int = 320,
):
    """
    Main inference endpoint.

    The browser captures a single video frame as a JPEG blob and POSTs it
    here together with the user's text query.  The server runs the full
    pipeline and returns JSON (no streaming).

    Body (multipart/form-data):
        frame  — JPEG file
        query  — plain-text question
        imgsz  — YOLO input size (default 320)
    """
    if not MODELS["loaded"]:
        raise HTTPException(status_code=503, detail="Models not loaded yet — please wait.")

    raw = await frame.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty frame upload.")

    try:
        frame_bgr = bytes_to_bgr(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info(f"📸 Analyse request: imgsz={imgsz}, query='{query}'")

    try:
        result = run_inference_on_frame(frame_bgr, query, imgsz)
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    logger.info(f"✅ Done — drivable={result['drivable_pct']}%, objects={result['detections']}")
    return result


# ─────────────────────────────────────────────
# Serve the single-file frontend
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "index_1.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>index_1.html not found — place it next to main.py</h1>",
                            status_code=404)
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ─────────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)