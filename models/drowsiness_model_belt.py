
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

from pylsl import StreamInlet, resolve_stream
from pylsl import resolve_streams
import pyhrv
import biosppy

import collections

seed = 1234

# Belt function
def get_biometrics(gen_inlet, rr_inlet, show_window=False, show_result=False):
    gen_sample, gen_timestamp = gen_inlet.pull_sample()
    rr_sample, rr_timstamp = rr_inlet.pull_sample()
    rr = rr_sample[0]
    return np.array([rr])

# Get the stream
streams = resolve_streams()
generalStream = None
rrStream = None
for stream in streams:
    if stream.name() == 'ZephyrGeneral':
        generalStream = stream
    elif stream.name() == 'ZephyrRtoR':
        rrStream = stream
general_inlet = StreamInlet(generalStream)
rr_inlet = StreamInlet(rrStream)

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

zeros = [923] * 1
rtor = collections.deque(zeros, 20)
while True:
    live = get_biometrics(general_inlet, rr_inlet, show_window=True, show_result=True)
    print("RR", live[0])

    if(abs(live[0]) != list(rtor)[-1]):
        rtor.append(abs(live[0]))
    rtor_list = list(rtor)
    sdnn = pyhrv.time_domain.sdnn(rtor_list)[0]
    rmssd = pyhrv.time_domain.rmssd(rtor_list)[0]
    mean_nn = pyhrv.time_domain.nni_parameters(rtor_list)[1]
    pNN50 = pyhrv.time_domain.nn50(rtor_list)[1]
    pNN20 = pyhrv.time_domain.nn20(rtor_list)[1]
    drowsy_pred = nn_drowsy.predict(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))
    drowsy_conf = nn_drowsy.predict_proba(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))
    print(rtor)
    print([sdnn,rmssd,mean_nn,pNN50,pNN20])
    print("D Pred", drowsy_pred)
    print("D Conf", drowsy_conf)
    print("D Classes", nn_drowsy.classes_)
