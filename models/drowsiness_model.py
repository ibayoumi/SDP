
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


# Drowsy dataset
data_drowsy = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_drowsy_data.csv")
data_drowsy = data_drowsy.to_numpy()

# Drowsy training/testing split
Xd, yd = data_drowsy[:,1:6], data_drowsy[:,-1]
Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.15, random_state=seed, shuffle=True)

# Drowsy NN model
nn_drowsy = MLPClassifier(learning_rate_init=0.0006, hidden_layer_sizes=(200,100,2), activation='logistic', solver='adam', learning_rate='constant', early_stopping=False, max_iter=20000, n_iter_no_change=400, random_state=seed)
nn_drowsy.fit(Xd_train, yd_train)

# Drowsy get training error
d_train_pred = nn_drowsy.predict(Xd_train)
d_train_error = 1 - (accuracy_score(yd_train, d_train_pred))

# Drowsy get testing error
d_test_pred = nn_drowsy.predict(Xd_test)
d_test_error = 1 - (accuracy_score(yd_test, d_test_pred))

print("Drowsy Training Error: ", d_train_error)
print("Drowsy Testing Error: ", d_test_error)

# Drowsy plot loss over epoch
plt.plot(range(nn_drowsy.n_iter_), nn_drowsy.loss_curve_)
plt.xlabel("epoch")
plt.ylabel("loss") 
plt.show()
