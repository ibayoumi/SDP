# csv_data_analyzer: used to observe data statistics (such as mean, ranges, and correlations) of each recorded label

import pandas as pd
import numpy as np


# Reading csv file
data = pd.read_csv("C:\\Users\\citla\\SeniorDesign\\Datasets\\citlali_control_10_27.csv")
#data = pd.read_csv("C:\\Users\\citla\\SeniorDesign\\Datasets\\ismail_control_10_24.csv")

# Ignore unamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Converting column data to list
HR = data['Heartrate'].tolist()

# print correlations (+-1 = Highest correlation)
print(data.loc[:, 'Heartrate':'Breathing Rate Amplitude'].corr())

# Turn csv columns into numpy arrays
    # Syntax: arr = np.array([<the array>])
labels = data.columns
for label in labels:
    col = np.array(data.loc[:, label])
    mean = np.mean(col)
    max = np.max(col)
    min = np.min(col)
    median = np.median(col)
    std = np.std(col)

    print(f"Data Report: \n Label = {label} \n Mean = {mean} \n Max = {max} \n Min = {min} \n Median = {median} \n Std = {std}\n")

