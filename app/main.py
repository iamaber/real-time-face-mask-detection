import io
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.utils import get_model, predict, preprocess_image

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_PIXELS = 40_000_000

# Decompression-bomb guard; oversized images raise instead of exhausting memory.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = get_model()
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict/")
def predict_mask(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}",
        )

    contents = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte limit",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        Image.open(io.BytesIO(contents)).verify()
        # verify() leaves the buffer unusable, so decode from a fresh handle.
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    image_tensor = preprocess_image(image)
    return predict(state["model"], image_tensor)
