
# Trying the pylsl example now to *hopefully* get real-time data
from pylsl import StreamInlet, resolve_stream


def main():
    try:
        # first resolve an EEG stream on the lab network
        print("looking for an EEG stream...")
        streams = resolve_stream('name', 'ZephyrGeneral')

        # create a new inlet to read from the stream
        inlet = StreamInlet(streams[0])

        # Create a new file
        filename = input("Enter CSV file name: ")
        file = open(f"{filename}.csv", "w")

        # Write labels to the file
        file.write("Time,Timestamp,Heartrate,Breathing Rate,Skin Temperature,Posture,Activity,Peak Acceleration,Battery Voltage,Breathing Rate Amplitude,ECG Amplitude,\
        ECG Noise,Vertical Acceleration Minimum,Vertical Acceleration Peak,Lateral Acceleration Minimum,Lateral Acceleration Peak,Sagittal Acceleration Minimum,\
        Sagittal Acceleration Peak,\n")

        while True:
            # get a new sample (you can also omit the timestamp part if you're not
            # interested in it)
            sample, timestamp = inlet.pull_sample()
            print(timestamp, sample)

            for item in sample[0:18]:
                file.write(str(item) + ',')
            file.write('\n')

    except KeyboardInterrupt:
        print("Keyboard Interrupt. Shutting down...")
        file.close()

    exit()
    

main()
# if __name__ == '__main__':
#     main()
