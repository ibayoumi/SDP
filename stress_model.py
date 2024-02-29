
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

seed = 1234


# Stress dataset
data_stress = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_stress_data_no_ecg.csv")
data_stress = data_stress.to_numpy()

# Stress training/testing split
Xs, ys = data_stress[:,1:3], data_stress[:,-1]
Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.3, random_state=seed, shuffle=True)

# Stress NN model
nn_stress = MLPClassifier(learning_rate_init=0.01, hidden_layer_sizes=(100,100), activation='logistic', solver='adam', learning_rate='constant', early_stopping=True, max_iter=200, n_iter_no_change=10, random_state=seed)
nn_stress.fit(Xs_train, ys_train)

# Stress get training error
s_train_pred = nn_stress.predict(Xs_train)
s_train_error = 1 - (accuracy_score(ys_train, s_train_pred))

# Stress get testing error
s_test_pred = nn_stress.predict(Xs_test)
s_test_error = 1 - (accuracy_score(ys_test, s_test_pred))

print("Stress Training Error: ", s_train_error)
print("Stress Testing Error: ", s_test_error)

# Stress plot loss over epoch
plt.plot(range(nn_stress.n_iter_), nn_stress.loss_curve_)
plt.xlabel("epoch")
plt.ylabel("loss") 
plt.show()