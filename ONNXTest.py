#FIXME 

import onnxruntime as ort
import pandas as pd
import numpy as np

onnx_path = "lightgbm_model.onnx"
session = ort.InferenceSession(onnx_path)
print('Model loaded...')

input_data = pd.read_csv(r'datasets\imputed_train_data.csv', nrows=10)
input_data = input_data.drop(columns=["isFraud"])
input_array = np.array(input_data, dtype=np.float32)

input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: input_array})

fraud_probabilities = outputs[1] 
i = 1
for fruad_probability in fraud_probabilities:
    prob_value = ((fruad_probability[0]) * 100)
    if prob_value > 70:
        print(f'{i}: Probability value: {prob_value:.6f}% (LIKELY FRUAD)')
    else: 
        print(f'{i}: Probability value: {prob_value:.6f}%')
    i += 1

print(fraud_probabilities)