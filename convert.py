import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load your saved model
model = joblib.load("model.pkl")

# 7 input features
initial_type = [("float_input", FloatTensorType([None, 7]))]

# Convert to ONNX
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save file
with open("aqi_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Done. ONNX model created.")