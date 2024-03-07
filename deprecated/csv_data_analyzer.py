# csv_data_analyzer: used to observe data statistics (such as mean, ranges, and correlations) of each recorded label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configure pandas display
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
#pd.set_option('display.width', 1000)


# Reading csv file
data = pd.read_csv("C:\\Users\\citla\\SeniorDesign\\Datasets\\citlali_control_10_27.csv")
#data = pd.read_csv("C:\\Users\\citla\\SeniorDesign\\Datasets\\ismail_lifting_10_28.csv")

# Ignore unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Converting column data to list
HR = data['Heartrate'].tolist()

# print correlations (+-1 = Highest correlation)
print(data.loc[:, 'Heartrate':'Breathing Rate Amplitude'].corr())

# Turn csv columns into numpy arrays
    # Syntax: arr = np.array([<the array>])
labels = data.columns
#fig, axes = plt.subplots(2, 7, figsize = (12, 12))
fig, axes = plt.subplots(1, 1, figsize = (12, 12))

# Indices for HR, BR, Posture, Activity, Peak Accel., ECG Amp. 
relevant_label_index = {2,3,5,6,7,10}
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple','tab:brown']
i = 0
axes_index = 0

for label in labels:
    col = np.array(data.loc[:, label])
    mean = np.mean(col)
    max = np.max(col)
    min = np.min(col)
    median = np.median(col)
    std = np.std(col)
    var = np.var(col) 
    
    # Plot values for each label
    if i in relevant_label_index:
        # Histograms
        # axes[0,axes_index].hist(col)
        # axes[0,axes_index].title.set_text(label)

        # axes[1,axes_index].plot(col, color='r')
        # axes[1,axes_index].title.set_text(label)

        # Line Graph
        axes.plot(col, color=colors[axes_index], label = label)
        axes_index += 1
        

    i += 1

    print(f"Data Report: \n Label = {label} \n Mean = {mean} \n Max = {max} \n Min = {min} \n Median = {median} \n Std = {std}\n Var = {var}\n")

plt.legend()
plt.show()

