# KF, HPF
from matplotlib import pyplot as plt
import numpy as np
from filter_class import *

# ==========     Parameter    ==========
# 측정해서 넣어야 한다.
# Noise of sensor
sensor_noise = 0.5  # 센서의 측정값의 표준편차
# sampling duration
sampling_duration = 0.01
# display delay
display_delay = 0.2

# ==========  Hyper parameters in KF, HF   ==========
# 1. KF
# sampling_duration동안의 "가속도의 변화율"의 표준편차
sigma_jerk = 2

# 몇 개의 time step뒤의 변위값을 예측할지 결정.
t_predict = 12

# 2. HF
# High Pass filter의 pass band를 결정하는 상수
RC = 1

# ==========    Sample data    ==========
# noise를 포함한 파형
print("Enter Sampling Number : ")
n = int(input())

amp = 3
x_data = amp * np.sin(np.array(range(n)) / 100)
#ori_data = amp * np.sin(np.array(range(n)) / 100)

file = open("data_2000.txt", "r")
line = file.readline()

i = 0

while line:
  if line:
      line = line.strip()
      a_x = int(line.split('	')[0])
      a_y = int(line.split('	')[1])
      scale_percent = int(line.split('	')[2])
      x_data[i] = a_x
      #x_data[i] = a_y

      i += 1

  line = file.readline()

# sensor_noise 만큼의 표준편차를 가진 노이즈를 추가
#noise_data = x_data+np.random.normal(0,sensor_noise,n)
noise_data = x_data

# ========== Kalman filter, KF ==========
kf_input = noise_data
kf_predict = []
kf_result = []

pos = 0
vel = -amp
acc = 0
x = np.array([[pos], [vel], [acc]])

# A, (const) state transition matrix
A = np.array([[1, sampling_duration, sampling_duration ** 2 / 2], \
              [0, 1, sampling_duration], [0, 0, 1]])
# P, (initial) covariance matrix of state vector
P = np.array([[100 ** 4 * sensor_noise ** 2, 0, 0], \
              [0, 100 ** 2 * sensor_noise ** 2, 0], [0, 0, sensor_noise ** 2]])
# R, (const) covariance of observation matrix
R = np.array([[sensor_noise ** 2]])
# Q, (const) covarience of process noise matrix
B = np.array([[sampling_duration ** 3 / 6], [sampling_duration ** 2 / 2], [sampling_duration]])
Q = np.dot(B, np.transpose(B)) * sigma_jerk ** 2
# H, (const) observation matrix
H = np.array([[0, 0, 1]])

filter_kf = Kalman(A, H, R, Q, P, x, t_predict)

for a in kf_input:
    result = filter_kf.update(a)
    kf_result.append(result[0])
    kf_predict.append(result[1])

# KF Result
plt.figure(figsize=(6, 4))
plt.title('Total ' + str(n) + ' time steps, ' + 'Predict ' + str(t_predict) + ' time steps using KF\n')
plt.plot(kf_input, label='Input Data')
plt.plot(kf_result, color='r', label='Pos. (KF result)')
plt.plot(kf_predict, color='g', label='Pos. (KF Predicted)')
plt.ylabel("Pos.")
plt.xlabel("Time")
plt.grid(True)
plt.legend(fontsize='x-small')

# ========== High Pass Filter, HPF ==========
hf_input = kf_predict
hf_output = []

filter_hf = HPF(RC, sampling_duration)

for x in hf_input:
    hf_output.append(filter_hf.update(x))

# HF result
plt.figure(figsize=(6, 5))
plt.subplot(2, 1, 1)
plt.title('HPF with RC = ' + str(RC) + '\n')
plt.plot([0, n], [0, 0], color=(0, 0, 0))
plt.plot(kf_input)
plt.ylabel("Before HF")
plt.grid(True)
plt.plot(hf_input, color='r')

plt.subplot(2, 1, 2)
plt.plot([0, n], [0, 0], color=(0, 0, 0))
plt.plot(kf_input)
plt.ylabel("After HF")
plt.xlabel("Time")
plt.grid(True)
plt.plot(hf_output, color='g')
plt.show()