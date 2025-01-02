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

input_data = pd.read_csv(r'datasets\imputed_train_data.csv', nrows=590540)
input_data = input_data.drop(columns=["isFraud"])

input_data = scaler.transform(input_data)  # ensures this scales the same way as during training
input_array = np.array(input_data, dtype=np.float32)

input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: input_array})

fraud_probabilities = outputs[1] 


def print_fruad_probabilities(probabilties, confidence_low = 50, confidence_high = 90, print_all = True):
    i = 1
    num_fruad_high = 0
    num_fruad_low = 0
    confidence_low = 50

    if (print_all):
        for probability in probabilties:
            prob_value = ((probability[1]) * 100)
            if prob_value > confidence_high:
                print(f'{i}: Probability value: {prob_value:.6f}% (FRUAD [HIGH])')
                num_fruad_high += 1
            elif prob_value > confidence_low:
                print(f'{i}: Probability value: {prob_value:.6f}% (FRUAD [LOW])')
                num_fruad_low += 1
            else: 
                print(f'{i}: Probability value: {prob_value:.6f}%')
            i += 1
    else:
        for probability in probabilties:
            prob_value = ((probability[1]) * 100)
            if prob_value > confidence_high:
                print(f'{i}: Probability value: {prob_value:.6f}% (FRUAD [HIGH])')
                num_fruad_high += 1
            elif prob_value >= confidence_low:
                print(f'{i}: Probability value: {prob_value:.6f}% (FRUAD [LOW])')
                num_fruad_low += 1
            i += 1
    print(f'High Confidence Fruadulant Transactions: {num_fruad_high} {num_fruad_high / 500000}')
    print(f'Low Confidence Fruadulant Transactions: {num_fruad_low} {num_fruad_low / 500000}')

print_fruad_probabilities(fraud_probabilities, print_all = False)
