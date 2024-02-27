import matplotlib.pyplot as plt
import numpy as np

"""
with open("data_log.csv", 'r') as file:
	lines = file.readlines()[1:]

matrix = []
for index in range(7):
	matrix.append(np.array([row.split(',')[index] for row in lines]))
print(np.array(matrix))
"""

arr = np.loadtxt("data_log_i1.csv", delimiter=",", dtype=str)
print(arr)


timestamp = []
speed = []
throttle = []
brake = []
steer = []
attrafficlight = []
heartrate = []
breathingrate = []
stress = []
drowsy = []
prev_time = 0

for row in arr[1:]:
	#if row[0] != prev_time:
		timestamp.append(float(row[0].split(':')[1] + '.' + row[0].split(':')[2]))
		speed.append(float(row[1]))
		throttle.append(float(row[2]))
		brake.append(float(row[3]))
		steer.append(float(row[4]))
		attrafficlight.append(float(row[5]))
		heartrate.append(float(row[6]))
		breathingrate.append(float(row[7]))
		stress.append(float(row[8]))
		drowsy.append(float(row[9]))
	#prev_time = row[0]


"""
timestamp = arr[1:,0]
speed = arr[1:,1]
throttle = arr[1:,2]
brake = arr[1:,3]
steer = arr[1:,4]
attrafficlight = arr[1:,5]
heartrate = arr[1:,6]
"""

fig, axes = plt.subplots(4, 1, figsize=(12, 3))
axes[0].plot(timestamp, heartrate, timestamp, breathingrate)
axes[0].legend(['heartrate', 'breathingrate'])
axes[1].plot(timestamp, speed)
axes[1].legend(['speed'])
axes[2].plot(timestamp, throttle, timestamp, brake, timestamp, steer, timestamp, attrafficlight)
axes[2].legend(['throttle', 'brake', 'steer', 'attrafficlight'])
axes[3].plot(timestamp, stress, timestamp, drowsy)
axes[3].legend(['stress', 'drowsy'])
plt.show()