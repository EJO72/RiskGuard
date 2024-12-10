# add imports
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import seaborn as sns
import matplotlib as plt

test_transactions = pd.read_csv(r'datasets/test_transaction.csv')
test_identity = pd.read_csv(r'datasets/test_identity.csv')
train_transactions = pd.read_csv(r'datasets/train_transaction.csv')
train_identity = pd.read_csv(r'datasets/train_identity.csv')

# merge train and test datasets independently on 'TransactionID'
train_data = train_transactions.merge(train_identity, on='TransactionID', how='left')
test_data = test_transactions.merge(test_identity, on='TransactionID', how='left')

# delete unused objects to save memory
del test_transactions, train_transactions, train_identity, test_identity

print("Train Data Overview:")
print(train_data.head(7))
# train data pretty much looks the same

# heatmap for the is fruad
plt.figure(figsize=(6, 4))
sns.countplot(x='isFraud', data=train_data)
plt.title("Fraud vs Non-Fraud Distribution")
plt.show()