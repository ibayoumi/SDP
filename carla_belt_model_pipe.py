
# PYTHON LIBRARIES
from multiprocessing import Process,Pipe
import random
from time import sleep, time
import sys
import collections
import pickle

# SKLEARN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# BIOHARNESS
from pylsl import StreamInlet, resolve_stream
from pylsl import resolve_streams
import pyhrv
import biosppy

# OSC
from osc4py3.as_eventloop import*
from osc4py3 import oscmethod as osm
import logging
import threading

# NUMPY
import numpy as np

# PANDAS
import pandas as pd

# MATPLOTLIB
import matplotlib.pyplot as plt

# CONSTANTS
seed = 1234

# GLOBALS
gsr_val = 0



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
            [hr, br, rr]
        """
        gen_sample, gen_timestamp = gen_inlet.pull_sample()
        rr_sample, rr_timstamp = rr_inlet.pull_sample()
        hr = gen_sample[2]
        br = gen_sample[3]
        rr = rr_sample[0]
        return np.array([hr, br, rr])

 
    def handlerfunction(*args):
        for arg in args:
            global gsr_val
            gsr_val = arg

    osc_startup()
    IP = '172.20.10.4'
    PORT = 8000
    osc_udp_client(IP, PORT, "udplisten")
    osc_udp_server(IP, PORT, "udpclient")

    # Associate Python functions with message address patterns, using defaults argument
    osc_method("/edaMikroS", handlerfunction, argscheme=osm.OSCARG_DATAUNPACK)


    # Resolve streams
    print("looking for an EEG stream...")
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


    # ---- Stress ML Model ----
    f = open("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/stress_pickle", "rb")
    nn_stress = pickle.load(f)
    f.close()
 

    # ---- Drowsy ML Model ----
    f = open("D:/SDP_Biometric_ADAS_CARLA_0.9.14/WindowsNoEditor/PythonAPI/examples/App_Zephyr_main/drowsy_model.pickle", "rb")
    nn_drowsy = pickle.load(f)
    f.close()
 

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
    plt.show()


    print("Intializing zephyrGeneral")
    msg = "Zephyr stream is working!!!!"
    child_conn.send(msg)
    print("PAST STREAMS")

    stress_conf = [0, 0]
    drowsy_conf = [0, 0]
    rtor_queue = collections.deque([923], 60)
    prev_value = 0
    # Pull samples from BioHarness and predict human state using ML Models
    while True:
        if child_conn.poll():
            child_conn.close()
            break

        # ---- Stress Predictions ----
        live = get_biometrics(general_inlet, rr_inlet, show_window=True, show_result=True)
        osc_process()

        live_stress = list(live[:2])
        live_stress.append(gsr_val)
        live_stress = np.array(live_stress)
        hr = str(live_stress[0])
        br = str(live_stress[1])
        gsr = str(live_stress[2])
        
        # Normalize heart rate and breathing rate values
        live_stress[0] = (live_stress[0]-25)/(240-25)
        live_stress[1] = (live_stress[1]-4)/(70-4)
        live_stress[2] = (live_stress[2]-1)/(20-1)

        # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"
        stress_pred = nn_stress.predict(live_stress.reshape(1,-1))
        stress_conf = nn_stress.predict_proba(live_stress.reshape(1,-1))
        print("HR", hr)
        print("BR", br)
        print("GSR", gsr)
        print("S Pred", stress_pred)
        print("S Conf", stress_conf)
        print("S Classes", nn_stress.classes_)
        print()


        # ---- Drowsy Predictions ----
        live_drowsy = live[2]
        rtor = str(abs(live_drowsy))

        if(live_drowsy != prev_value):
            prev_value = live_drowsy
            rtor_queue.append(abs(live_drowsy))
        rtor_list = list(rtor_queue)
        sdnn = pyhrv.time_domain.sdnn(rtor_list)[0]
        rmssd = pyhrv.time_domain.rmssd(rtor_list)[0]
        mean_nn = pyhrv.time_domain.nni_parameters(rtor_list)[1]
        pNN50 = pyhrv.time_domain.nn50(rtor_list)[1]/100
        pNN20 = pyhrv.time_domain.nn20(rtor_list)[1]/100

        # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"
        drowsy_pred = nn_drowsy.predict(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))
        drowsy_conf = nn_drowsy.predict_proba(np.array([sdnn,rmssd,mean_nn,pNN50,pNN20]).reshape(1,-1))
        print("R to R", rtor_queue)
        print("Array", [sdnn,rmssd,mean_nn,pNN50,pNN20])
        print("D Pred", drowsy_pred)
        print("D Conf", drowsy_conf)
        print("D Classes", nn_drowsy.classes_)
        print()

        # Send data to CARLA
        msg = [hr, br, gsr, rtor, stress_conf[0][1], drowsy_conf[0][1]]
        print(msg)
        print()
        child_conn.send(msg)

    osc_terminate()