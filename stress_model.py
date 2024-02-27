# IMPORTS

# SKLEARN
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# BIOHARNESS/BIOMETRICS
from pylsl import StreamInlet, resolve_stream, resolve_streams
import pyhrv
import biosppy

# NUMPY
import numpy as np

# PANDAS
import pandas as pd

# TIME
from time import time

# CONSTANTS
seed = 1234


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
    hr = gen_sample[2]
    br = gen_sample[3]

    """

    # Convert HR samples into SDNN, RMSSD, and mean HR
    hr_window = []
    rr_window = []
    # hr_window = [68.0, 70.0, 69.0, 68.0, 67.0, 64.0, 62.0, 63.0, 62.0, 61.0, 63.0, 69.0, 71.0, 70.0, 67.0, 66.0, 64.0, 63.0, 64.0, 63.0, 65.0, 67.0, 67.0, 68.0, 68.0, 66.0, 64.0, 65.0, 64.0, 65.0, 67.0]
    # rr_window = [0.872, 0.872, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, -0.893, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, 0.868, -0.8220000000000001, -0.8220000000000001]
    # Can either use time() or belt's timestamp. Don't know which would be best.
    init_time = time()
    final_time = time()

    # Take 30s intervals
    while (final_time - init_time < 30):
        gen_sample, gen_timestamp = gen_inlet.pull_sample()
        rr_sample, rr_timestamp = rr_inlet.pull_sample()

        hr_window.append(gen_sample[2])
        rr_window.append(rr_sample[0] * (10**-3))

        final_time = time()

    if(show_window):
        print("HR Window: ", hr_window)
        print("RR Window: ", rr_window)

    # Calculate sdnn, rmssd
    sdnn = pyhrv.time_domain.sdnn(rr_window)['sdnn'] * (10**-2)  # unsure of this conversion...
    rmssd = pyhrv.time_domain.rmssd(rr_window)['rmssd'] * (10**-2)
    mean_hr = np.mean(hr_window)

    if(show_result):
        print("SDNN: ", sdnn)
        print("RMSSD: ", rmssd)
        print("MEAN HR: ", mean_hr)
        #print("PYHRV MEAN HR: ", pyhrv.time_domain.hr_parameters(rr_window[10:12])['hr_mean'] * (10**3))
    
    return np.array([sdnn, rmssd, mean_hr])


    """

    return np.array([hr, br])


# Read in data and convert to numpy array
data = pd.read_csv("C:\\Users\\citla\\SeniorDesign\\Datasets\\external_data\\MIT_driver_stress\\extracted_stress_data_no_ecg.csv")
data = data.to_numpy()

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


# Train ML Model

# Data Splits
# Split into X and Y
X, y = data[:,1:3], data[:,-1]
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)

# Initialize the model
# dt = DecisionTreeClassifier(criterion='entropy', max_depth=15, min_samples_split=200, min_samples_leaf=50)
dt = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=45, min_samples_split=50, min_samples_leaf=50, max_leaf_nodes=150)

# Fit the model to the training set
dt.fit(X_tr, y_tr)

# Compute the training and test errors
y_tr_pred = dt.predict(X_tr)
train_error = 1 - (accuracy_score(y_tr, y_tr_pred))

y_val_pred = dt.predict(X_val)
val_error = 1 - (accuracy_score(y_val, y_val_pred))

print("Training Error: ", train_error)
print("Testing Error: ", val_error)

# Pull samples from BioHarness and predict human state using ML Model
# TODO: While loop that continuously pulls samples and predicts
while(True):
    X_te = get_biometrics(general_inlet, rr_inlet, show_window=True, show_result=True)

    y_te_pred = dt.predict(X_te.reshape(1,-1))  # Expects 2D array. "Reshape your data using array.reshape(1,-1) if it contains a single sample"
    print(y_te_pred)

    # Confidence Score
    conf = dt.predict_proba(X_te.reshape(1,-1))
    print(conf)
    print(dt.classes_)

    # TODO: Send data to CARLA (create function/module for this)

