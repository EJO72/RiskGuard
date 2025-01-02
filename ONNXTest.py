#FIXME 
import pickle
import onnxruntime as ort
import pandas as pd
import numpy as np

# Load the scaler
scaler_path = "scaler.pkl"  # Path to the scaler saved during training
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

onnx_path = "lightgbm_model.onnx"
session = ort.InferenceSession(onnx_path)
print('Model loaded...')

input_data = pd.read_csv(r'datasets\imputed_train_data.csv', nrows=300)
input_data = input_data.drop(columns=["isFraud"])

input_data = scaler.transform(input_data)  # ensures this scales the same way as during training
input_array = np.array(input_data, dtype=np.float32)

input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: input_array})

fraud_probabilities = outputs[1] 
i = 1
for fruad_probability in fraud_probabilities:
    prob_value = ((fruad_probability[1]) * 100)
    if prob_value > 70:
        print(f'{i}: Probability value: {prob_value:.6f}% (LIKELY FRUAD)')
    else: 
        print(f'{i}: Probability value: {prob_value:.6f}%')
    i += 1

print(fraud_probabilities)