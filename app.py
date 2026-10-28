import io
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/Baseline.onnx")
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "baseline-fp32-v1"
)

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# ---------------------------------------------------------
# Load ONNX model once when the application starts
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model does not exist: {MODEL_PATH}"
    )

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

INPUT_NAME = input_info.name
OUTPUT_NAME = output_info.name


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="EML MNIST Edge-AI Service",
    version=MODEL_VERSION
)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

def preprocess_image(image: Image.Image) -> np.ndarray:

    # MNIST is grayscale
    image = image.convert("L")

    # Exercise 7 model expects 28x28
    image = image.resize((28, 28))

    # Equivalent to ToTensor()
    x = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    # Same normalization as Exercise 7
    x = (x - MNIST_MEAN) / MNIST_STD

    # [28,28]
    #       ↓
    # [batch, channel, height, width]
    # [1,1,28,28]

    x = x[np.newaxis, np.newaxis, :, :]

    return x.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:

    logits = logits - np.max(
        logits,
        axis=1,
        keepdims=True
    )

    exp_values = np.exp(logits)

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_file": MODEL_PATH.name,
        "provider": session.get_providers()[0]
    }


# ---------------------------------------------------------
# Model information endpoint
# ---------------------------------------------------------

@app.get("/model")
def model_information():

    return {
        "model_version": MODEL_VERSION,
        "model_file": MODEL_PATH.name,

        "input": {
            "name": INPUT_NAME,
            "shape": input_info.shape,
            "type": input_info.type
        },

        "output": {
            "name": OUTPUT_NAME,
            "shape": output_info.shape,
            "type": output_info.type
        },

        "providers": session.get_providers()
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )

    raw_data = await file.read()

    try:

        image = Image.open(
            io.BytesIO(raw_data)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    x = preprocess_image(image)

    # -----------------------------------------------------
    # Measure inference time only
    # -----------------------------------------------------

    start = time.perf_counter()

    outputs = session.run(
        [OUTPUT_NAME],
        {
            INPUT_NAME: x
        }
    )

    latency_ms = (
        time.perf_counter() - start
    ) * 1000

    logits = np.asarray(
        outputs[0],
        dtype=np.float32
    )

    probabilities = softmax(logits)

    prediction = int(
        np.argmax(probabilities[0])
    )

    confidence = float(
        probabilities[0][prediction]
    )

    return {
        "prediction": prediction,
        "confidence": round(confidence, 6),
        "latency_ms": round(latency_ms, 3),
        "model_version": MODEL_VERSION,
        "model_file": MODEL_PATH.name
    }