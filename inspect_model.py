import onnxruntime as ort

MODEL_PATH = "models/Baseline.onnx"

print("=" * 60)
print("EML EDGE-AI — ONNX MODEL INSPECTION")
print("=" * 60)

print(f"\nLoading model: {MODEL_PATH}")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("\n[1] Available ONNX Runtime providers")
print(ort.get_available_providers())

print("\n[2] Active providers")
print(session.get_providers())

print("\n[3] Model inputs")

for inp in session.get_inputs():
    print(f"Name  : {inp.name}")
    print(f"Shape : {inp.shape}")
    print(f"Type  : {inp.type}")

print("\n[4] Model outputs")

for out in session.get_outputs():
    print(f"Name  : {out.name}")
    print(f"Shape : {out.shape}")
    print(f"Type  : {out.type}")

print("\n" + "=" * 60)
print("BASELINE ONNX LOADED SUCCESSFULLY")
print("=" * 60)