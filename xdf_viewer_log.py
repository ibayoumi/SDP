# import pyxdf
# import matplotlib.pyplot as plt
# import numpy as np

# # Second test: Respiration AND Heart Rate
# #data, header = pyxdf.load_xdf('C:\\Users\\citla\\SeniorDesign\\App-Zephyr-main\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-001_eeg.xdf')
# # First test: Respiration Rate
# #data, header = pyxdf.load_xdf('C:\\Users\\citla\\SeniorDesign\\App-Zephyr-main\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-003_eeg.xdf')
# # Third test: Heart rate while watching Analog Horror
# #data, header = pyxdf.load_xdf('C:\\Users\\citla\\SeniorDesign\\App-Zephyr-main\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default[_acq-]_run-001_eeg.xdf')
# data, header = pyxdf.load_xdf('C:\\Users\\citla\\SeniorDesign\\App-Zephyr-main\\sub-P001\\ses-S001\\eeg\\sub-P001_ses-S001_task-Default_run-007_eeg.xdf')

# for stream in data:
#     y = stream['time_series']

#     if isinstance(y, list):
#         # list of strings, draw one vertical line for each marker
#         for timestamp, marker in zip(stream['time_stamps'], y):
#             plt.axvline(x=timestamp)
#             print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
#     elif isinstance(y, np.ndarray):
#         # numeric data, draw as lines
#         print(y)
#         plt.plot(stream['time_stamps'], y)
#     else:
#         raise RuntimeError('Unknown stream format')

# plt.show()

# Trying the pylsl example now to *hopefully* get real-time data
from pylsl import StreamInlet, resolve_stream


def main():
    # first resolve an EEG stream on the lab network
    print("TEST: idk where this thing is stuck ;v;")
    print("looking for an EEG stream...")
    #streams = resolve_stream('type', 'EEG')
    streams = resolve_stream('name', 'ZephyrGeneral')

    print("PAST STREAMS")
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    print("PAST INLET")

    filename = input("Enter CSV file name: ")

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        print(timestamp, sample)

        with open(f"{filename}.csv", "a") as file:
            for item in sample:
                file.write(str(item) + ',')
            file.write('\n')


main()
# if __name__ == '__main__':
#     main()