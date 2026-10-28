from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_PATH = Path("models/Quant_8bit.onnx")
IMAGE_PATH = Path("test_digit.png")

EXPECTED_LABEL = 7

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def preprocess_image(image_path: Path) -> np.ndarray:
    image = Image.open(image_path)

    # Exercise 7 uses one grayscale channel.
    image = image.convert("L")

    # The ONNX model expects 28 x 28.
    image = image.resize((28, 28))

    # Equivalent to torchvision.transforms.ToTensor():
    # uint8 [0,255] -> float32 [0,1]
    x = np.asarray(image, dtype=np.float32) / 255.0

    # Same normalization used in Exercise 7:
    # Normalize((0.1307,), (0.3081,))
    x = (x - MNIST_MEAN) / MNIST_STD

    # Current shape:
    # [28, 28]
    #
    # Required ONNX shape:
    # [batch, channel, height, width]
    # [1, 1, 28, 28]
    x = x[np.newaxis, np.newaxis, :, :]

    return x.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    # Numerically stable softmax.
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


print("=" * 60)
print("EML EDGE-AI — LOCAL ONNX INFERENCE")
print("=" * 60)

print("\nModel :", MODEL_PATH)
print("Image :", IMAGE_PATH)
print("Truth :", EXPECTED_LABEL)

# -------------------------------------------------------
# Load model
# -------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]
output_info = session.get_outputs()[0]

print("\nONNX input")
print("Name :", input_info.name)
print("Shape:", input_info.shape)
print("Type :", input_info.type)

# -------------------------------------------------------
# Preprocess image
# -------------------------------------------------------

x = preprocess_image(IMAGE_PATH)

print("\nPreprocessed tensor")
print("Shape:", x.shape)
print("Type :", x.dtype)
print("Min  :", float(x.min()))
print("Max  :", float(x.max()))

# -------------------------------------------------------
# Run inference
# -------------------------------------------------------

outputs = session.run(
    [output_info.name],
    {
        input_info.name: x
    }
)

logits = outputs[0][0]

# Exercise 7 ends with Linear(..., 10), so these are
# class scores/logits, not probabilities.
probabilities = softmax(logits)

prediction = int(np.argmax(probabilities))
confidence = float(probabilities[prediction])

# -------------------------------------------------------
# Print class probabilities
# -------------------------------------------------------

print("\nDigit probabilities")

for digit, probability in enumerate(probabilities):
    print(
        f"{digit}: "
        f"{probability * 100:6.2f}% "
        f"logit={logits[digit]:8.4f}"
    )

print("\n" + "=" * 60)

print("TRUE LABEL :", EXPECTED_LABEL)
print("PREDICTION :", prediction)
print("CONFIDENCE :", f"{confidence * 100:.2f}%")

if prediction == EXPECTED_LABEL:
    print("RESULT     : CORRECT ✅")
else:
    print("RESULT     : INCORRECT ❌")

print("=" * 60)