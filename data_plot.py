import matplotlib.pyplot as plt
import numpy as np

arr = np.loadtxt("data_log_i2.csv", delimiter=",", dtype=str)
print(arr)

timestamp = []
speed = []
throttle = []
brake = []
steer = []
attrafficlight = []
heartrate = []
breathingrate = []
gsr = []
rtor = []
stress = []
drowsy = []
prev_time = 0

for row in arr[1:]:
	timestamp.append(float(row[0].split(':')[1] + '.' + row[0].split(':')[2]))
	speed.append(float(row[1]))
	throttle.append(float(row[2]))
	brake.append(float(row[3]))
	steer.append(float(row[4]))
	attrafficlight.append(float(row[5]))
	heartrate.append(float(row[6]))
	breathingrate.append(float(row[7]))
	gsr.append(float(row[8]))
	rtor.append(float(row[9]))
	stress.append(float(row[10]))
	drowsy.append(float(row[11]))

fig, axes = plt.subplots(4, 1, figsize=(12, 3))
axes[0].plot(timestamp, heartrate, timestamp, breathingrate, timestamp, gsr)
axes[0].legend(['heartrate', 'breathingrate', 'gsr'])
#axes[1].plot(timestamp, rtor)
#axes[1].legend(['rtor'])
axes[1].plot(timestamp, speed)
axes[1].legend(['speed'])
axes[2].plot(timestamp, throttle, timestamp, brake, timestamp, steer, timestamp, attrafficlight)
axes[2].legend(['throttle', 'brake', 'steer', 'attrafficlight'])
#axes[3].plot(timestamp, stress, timestamp, drowsy)
#axes[3].legend(['stress', 'drowsy'])
axes[3].plot(timestamp, stress)
axes[3].legend(['stress'])
plt.show()