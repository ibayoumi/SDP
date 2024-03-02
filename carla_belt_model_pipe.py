from pylsl import StreamInlet, resolve_stream
from multiprocessing import Process,Pipe
import random
from time import sleep, time
import sys

# IMPORTS
import collections

# SKLEARN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# BIOHARNESS/BIOMETRICS
from pylsl import resolve_streams
import pyhrv
import biosppy

# NUMPY
import numpy as np

# PANDAS
import pandas as pd

import matplotlib.pyplot as plt

# CONSTANTS
seed = 1234


def start_stream(child_conn):

    def get_biometrics(gen_inlet, rr_inlet, show_window=False, show_result=False):
        """
        Function that extracts HRV features from BioHarness data. The HRV
        features are calculated over a 30s interval.
    
        Parameters
        ----------
            gen_inlet : BioHarness general inlet (created from ZephyrGeneral stream)
            rr_inlet  : BioHarness rr inlet (created from ZephyrRtoR stream)
            show_window : If True, print values in HR and RR window
            show_result : If True, print calculated SDNN, RMSSD, and mean_hr values
        
        Returns
        -------
            A NumPy array of a list in the following form:
            [sdnn, rmsdd, mean_hr]
        """
        gen_sample, gen_timestamp = gen_inlet.pull_sample()
        rr_sample, rr_timstamp = rr_inlet.pull_sample()
        hr = gen_sample[2]
        br = gen_sample[3]
        rr = rr_sample[0]
        return np.array([hr, br, rr])


    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_streams()
    generalStream = None
    rrStream = None

    for stream in streams:
        if stream.name() == 'ZephyrGeneral':
            generalStream = stream
        elif stream.name() == 'ZephyrRtoR':
            rrStream = stream

    # create a new inlet to read from the stream
    general_inlet = StreamInlet(generalStream)
    rr_inlet = StreamInlet(rrStream)


    # ---- Stress ML Model ----

    # Stress dataset
    data_stress = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_stress_data_no_ecg.csv")
    data_stress = data_stress.to_numpy()

    # Stress training/testing split
    Xs, ys = data_stress[:,1:3], data_stress[:,-1]
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.5, random_state=seed, shuffle=True)

    # Stress NN model
    nn_stress = MLPClassifier(learning_rate_init=0.001, hidden_layer_sizes=(70,14), activation='logistic', solver='adam', learning_rate='constant', early_stopping=False, max_iter=1000, n_iter_no_change=100, random_state=seed)
    nn_stress.fit(Xs_train, ys_train)

    # Stress get training error
    s_train_pred = nn_stress.predict(Xs_train)
    s_train_error = 1 - (accuracy_score(ys_train, s_train_pred))

    # Stress get testing error
    s_test_pred = nn_stress.predict(Xs_test)
    s_test_error = 1 - (accuracy_score(ys_test, s_test_pred))

    print("Stress Training Error: ", s_train_error)
    print("Stress Testing Error: ", s_test_error)


    # ---- Drowsy ML Model ----

    # Drowsy dataset
    data_drowsy = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_drowsy_data.csv")
    data_drowsy = data_drowsy.to_numpy()

    # Drowsy training/testing split
    Xd, yd = data_drowsy[:,1:6], data_drowsy[:,-1]
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xd, yd, test_size=0.15, random_state=seed, shuffle=True)

    # Drowsy NN model
    nn_drowsy = MLPClassifier(learning_rate_init=0.0006, hidden_layer_sizes=(200,100,2), activation='logistic', solver='adam', learning_rate='constant', early_stopping=False, max_iter=1000, n_iter_no_change=200, random_state=seed)
    nn_drowsy.fit(Xd_train, yd_train)

    # Drowsy get training error
    d_train_pred = nn_drowsy.predict(Xd_train)
    d_train_error = 1 - (accuracy_score(yd_train, d_train_pred))

    # Drowsy get testing error
    d_test_pred = nn_drowsy.predict(Xd_test)
    d_test_error = 1 - (accuracy_score(yd_test, d_test_pred))

    print("Drowsy Training Error: ", d_train_error)
    print("Drowsy Testing Error: ", d_test_error)


    # Plot Stress and Drowsy loss over epoch
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].set_title("Stress NN Model")
    axes[0].plot(range(nn_stress.n_iter_), nn_stress.loss_curve_)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Drowsy NN Model")
    axes[1].plot(range(nn_drowsy.n_iter_), nn_drowsy.loss_curve_)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    #plt.show()
    
    

    print("Intializing zephyrGeneral")
    #streams = resolve_stream('name', 'ZephyrGeneral')
    msg = "Zephyr stream is working!!!!"
    child_conn.send(msg)
    print("PAST STREAMS")
    #inlet = StreamInlet(streams[0])

    stress_conf = [0, 0]
    drowsy_conf = [0, 0]
    zeros = [100] * 50
    rtor = collections.deque(zeros, 50)
    # Pull samples from BioHarness and predict human state using ML Model
    # TODO: While loop that continuously pulls samples and predicts
    while True:
        if child_conn.poll():
            child_conn.close()
            break

        # ---- Stress Predictions ----
        Xd_te = get_biometrics(general_inlet, rr_inlet, show_window=True, show_result=True)
        Xs_te = Xd_te[:2]
        # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"
        stress_pred = nn_stress.predict(Xs_te.reshape(1,-1))

        # Stress Confidence Score
        stress_conf = nn_stress.predict_proba(Xs_te.reshape(1,-1))
        print("SP:", Xs_te)
        print("S Conf", stress_conf)
        print("S Classes", nn_stress.classes_)

        # ---- Drowsy Predictions ----
        if(abs(Xd_te[2]) != list(rtor)[-1]):
            rtor.append(abs(Xd_te[2]))
        rtor_list = list(rtor)
        sdnn = pyhrv.time_domain.sdnn(rtor_list)[0]
        rmssd = pyhrv.time_domain.rmssd(rtor_list)[0]
        mean_nn = pyhrv.time_domain.nni_parameters(rtor_list)[1]
        pNN50 = pyhrv.time_domain.nn50(rtor_list)[1]
        pNN20 = pyhrv.time_domain.nn20(rtor_list)[1]
        # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"
        drowsy_pred = nn_drowsy.predict(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))

        # Drowsy Confidence Score
        drowsy_conf = nn_drowsy.predict_proba(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))
        print("DP:", np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]))
        print("D Conf", drowsy_conf)
        print("D Classes", nn_drowsy.classes_)
        # Send data to CARLA
        sample, timestamp = general_inlet.pull_sample()
        #sleep(0.5)
        msg = [str(sample[2]), str(sample[3]), stress_conf[0][1], drowsy_conf[0][1], Xd_te[2]]
        print(msg)
        child_conn.send(msg)