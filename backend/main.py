from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError
import uvicorn

MODEL_PATH = Path(__file__).resolve().parent / "mnist_model.keras"
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the trained model once at startup instead of on the first request.
    global model
    if MODEL_PATH.exists():
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as error:
            print(f"Failed to load model from {MODEL_PATH}: {error}")
    else:
        print(f"Model not found at {MODEL_PATH}. Place your trained model there and restart.")
    yield


app = FastAPI(title="Handwritten Number Upload API", lifespan=lifespan)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(contents: bytes) -> np.ndarray:
    """Convert an uploaded image into the shape used by the MNIST model."""
    try:
        image = Image.open(BytesIO(contents)).convert("L")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.") from error

    image = ImageOps.autocontrast(image)

    # Detect background using corners: if corners are light, background is light -> invert
    # so digit becomes light on dark (what difference() expects for bbox detection).
    corners = [image.getpixel(p) for p in ((0, 0), (image.width - 1, 0), (0, image.height - 1), (image.width - 1, image.height - 1))]
    if sum(corners) / len(corners) > 127:
        image = ImageOps.invert(image)

    # Now image has dark background (0), light digit -> difference() finds digit bbox
    bounding_box = ImageChops.difference(image, Image.new("L", image.size, 0)).getbbox()
    if bounding_box:
        image = image.crop(bounding_box)

    image.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    offset = ((28 - image.width) // 2, (28 - image.height) // 2)
    canvas.paste(image, offset)
    return np.asarray(canvas, dtype=np.float32)[None, ...] / 255.0


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}. Run 'python train_model.py' first.",
            )
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "model_available": MODEL_PATH.exists()}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    input_image = preprocess_image(contents)
    try:
        predictions = get_model().predict(input_image, verbose=0)[0]
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Model inference failed.") from error
    prediction = int(np.argmax(predictions))

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(contents),
        "prediction": prediction,
        "confidence": round(float(predictions[prediction]), 4),
        "message": f"The predicted number is {prediction}.",
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
