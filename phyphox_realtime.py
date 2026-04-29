import requests
import time
import numpy as np
import matplotlib.pyplot as plt

from ahrs.filters import Madgwick
from scipy.spatial.transform import Rotation

#
# CONFIG
#

PP_ADDRESS = ""
WINDOW_SIZE = 200  # number of samples to display
SLEEP_TIME = 0.01  # polling delay

config = requests.get(PP_ADDRESS + "/config").json()
channels = [b["name"] for b in config["buffers"]]

url = PP_ADDRESS + "/get?" + "&".join(channels)

# Sensors
ACC = ['accX', 'accY', 'accZ']
GYR = ['gyroX', 'gyroY', 'gyroZ']
MAG = ['magX', 'magY', 'magZ']

# 
# Initializations
# 

madgwick = Madgwick(beta=0.1)  # Filter
quat = np.array([1., 0., 0., 0.])  # initial quaternion
angles_buffer = []  # Euler angles storage for plotting
prev_acc_time = None  # Timing of previously retrieved data

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
lines = ax.plot(np.zeros((WINDOW_SIZE, 3)))
ax.set_ylim(-180, 180)
ax.set_title("Real-time Orientation (Euler Angles)")
ax.set_xlabel("Samples")
ax.set_ylabel("Degrees")

# 
# Main Loop 
#
while True:
    try:
        # Request the data from Phyphox
        data = requests.get(url).json()
        #print(data)

        # Extract latest values
        values = {
            ch: data["buffer"][ch]["buffer"][0]
            for ch in channels
        }

        # Get one vector for each sensor
        acc = np.array([values[k] for k in ACC], dtype=float)
        gyr = np.array([values[k] for k in GYR], dtype=float)
        mag = np.array([values[k] for k in MAG], dtype=float)

        # Determine time since last update from accelerometer timestamps
        acc_time = data["buffer"]["acc_time"]["buffer"][0]
        if prev_acc_time is None:
            dt = 0.01  # fallback for first iteration
        else:
            dt = acc_time - prev_acc_time
        prev_acc_time = acc_time

        # Update the Madgwick filter
        quat = madgwick.updateMARG(q=quat, gyr=gyr, acc=acc, mag=mag, dt=dt)

        # quaternion to Euler angles
        euler  = Rotation.from_quat(quat, scalar_first=True).as_euler('zyx', degrees=True)

        # Store Euler Angles 
        angles_buffer.append(euler)
        if len(angles_buffer) > WINDOW_SIZE:
            angles_buffer.pop(0)  # Drop oldest samples, if window size is exceeded

        # Update plot
        data_np = np.array(angles_buffer)
        for i in range(3):
            lines[i].set_ydata(
                np.pad(data_np[:, i],
                       (WINDOW_SIZE - len(data_np), 0),
                       constant_values=np.nan)
            )

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(SLEEP_TIME)

    except Exception as e:
        print("Error:", e)
        time.sleep(1)