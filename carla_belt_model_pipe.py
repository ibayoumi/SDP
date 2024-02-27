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

# CONSTANTS
seed = 1234


def start_stream(child_conn):

    # FUNCTIONS
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


    
    # Read in data and convert to numpy array
    #data = pd.read_csv("extracted_stress_data.csv")
    data = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_stress_data_no_ecg.csv")
    data = data.to_numpy()

    data_drowsy = pd.read_csv("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/extracted_drowsy_data.csv")
    data_drowsy = data_drowsy.to_numpy()
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


    # ---- Train ML Models ----

    # ----  Train Stress Model  ----
    # Data Splits
    #Xs, ys = data[:,:3], data[:,3]    # Use this when GSR included
    Xs, ys = data[:,1:3], data[:,-1]
    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(Xs, ys, test_size=0.2, random_state=seed, shuffle=True)

    # Initialize the model
    #dt_stress = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_split=200, min_samples_leaf=50)  # this was the original model
    #dt_stress = DecisionTreeClassifier(criterion='entropy', max_depth=100, min_samples_split=10, min_samples_leaf=1, max_features='log2', ccp_alpha=0.0000001)
    rf_stress = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=1000, min_samples_split=200, min_samples_leaf=10, max_leaf_nodes=51000)

    # Fit the model to the training set
    rf_stress.fit(Xs_tr, ys_tr)

    # Compute the training and test errors
    ys_tr_pred = rf_stress.predict(Xs_tr)
    train_error_s = 1 - (accuracy_score(ys_tr, ys_tr_pred))

    ys_val_pred = rf_stress.predict(Xs_val)
    val_error_s = 1 - (accuracy_score(ys_val, ys_val_pred))

    print("Stress Training Error: ", train_error_s)
    print("Stress Testing Error: ", val_error_s)


    # ----  Train Drowsy ML model  ----
    Xd, yd = data_drowsy[:,:5], data_drowsy[:,-1]
    Xd_tr, Xd_val, yd_tr, yd_val = train_test_split(Xd, yd, test_size=0.25, random_state=seed, shuffle=True)

    # Initialize the model
    #dt_drowsy = DecisionTreeClassifier(criterion="entropy", splitter='best', max_depth=40, min_samples_split=50, min_samples_leaf=50, min_weight_fraction_leaf=0.0,
    #                       max_features=None, random_state=1234, max_leaf_nodes = 1000)
    #rf_drowsy = RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=1000, min_samples_split=200, min_samples_leaf=1, max_leaf_nodes=100000)
    rf_drowsy = MLPClassifier(learning_rate_init=0.01, hidden_layer_sizes=(300,200,50), activation='relu', solver='adam', learning_rate='constant')
    # Fit the model to the training set
    rf_drowsy.fit(Xd_tr, yd_tr)

    # Compute the training and test errors
    yd_tr_pred = rf_drowsy.predict(Xd_tr)
    train_error_d = 1 - (accuracy_score(yd_tr, yd_tr_pred))

    yd_val_pred = rf_drowsy.predict(Xd_val)
    val_error_d = 1 - (accuracy_score(yd_val, yd_val_pred))

    print("Drowsy Training Error: ", train_error_d)
    print("Drowsy Testing Error: ", val_error_d)

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
        ys_te_pred = rf_stress.predict(Xs_te.reshape(1,-1))  # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"

        # Confidence Score
        stress_conf = rf_stress.predict_proba(Xs_te.reshape(1,-1))
        print("SP:", Xs_te)
        #stress_conf = dt_stress.predict_proba(np.array([40, 5]).reshape(1,-1))
        print("S Conf", stress_conf)
        #print("S Classes", dt_stress.classes_)

        # ---- Drowsy Predictions ----
        #Xd_te = get_biometrics(general_inlet, rr_inlet, show_window=True, show_result=True)
        if(abs(Xd_te[2]) != list(rtor)[-1]):
            rtor.append(abs(Xd_te[2]))
        rtor_list = list(rtor)
        #print(rtor_list)
        sdnn = pyhrv.time_domain.sdnn(rtor_list)[0]
        rmssd = pyhrv.time_domain.rmssd(rtor_list)[0]
        mean_nn = pyhrv.time_domain.nni_parameters(rtor_list)[1]
        pNN50 = pyhrv.time_domain.nn50(rtor_list)[1]
        pNN20 = pyhrv.time_domain.nn20(rtor_list)[1]
        sdsd = pyhrv.time_domain.sdsd(rtor_list)[0]
        print("DrP:", np.array([sdnn,rmssd,mean_nn,pNN50,pNN20,sdsd]))
        yd_te_pred = rf_drowsy.predict(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20,sdsd]).reshape(1,-1))  # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"

        # Confidence Score
        drowsy_conf = rf_drowsy.predict_proba(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20,sdsd]).reshape(1,-1))
        print("D Conf", drowsy_conf)
        #print("D Classes", dt_drowsy.classes_)

        # Send data to CARLA (create function/module for this)    
        sample, timestamp = general_inlet.pull_sample()
        #sleep(0.5)
        msg = [str(sample[2]), str(sample[3]), stress_conf, drowsy_conf, Xd_te[2]]
        print(msg)
        child_conn.send(msg)