# KF, HPF
from matplotlib import pyplot as plt
import numpy as np


#==========     Parameter    ==========
# 측정해서 넣어야 한다.
# Noise of sensor
sensor_noise = 0.5 # 센서의 측정값의 표준편차
# sampling duration
sampling_duration = 0.01
# display delay
display_delay = 0.2





#==========  Hyper parameters in KF, HF   ==========
# 1. KF
# sampling_duration동안의 "가속도의 변화율"의 표준편차
sigma_jerk = 2
''' 간단히 설명하자면
x_next = x + x'*delta_t + x"*delta_t**2/2   (단, x'는 시간에 대한 미분)
이렇게 근사하는데, 고려하지 않는 항인
x'"*delta_t**3/6 + x""*delta_y**4/24 + ... 항의 값에 대한 고려이다.
시간에 대해 세 번 이상 미분한 값을 측정하지는 않으므로, 그 표준편차를 고려하는 것.
직접 모터위에 가속도센서를 올려서 측정하거나, 아니면 이론적으로 계산해서 넣는 것이 가장 좋지만,
값을 바꿔보면서 실험해봐도 좋다.
값이 클수록 과거의 데이터보다 현재 측정된 데이터에 영향을 크게 받는다는 점에 유의하자.
'''
# 몇 개의 time step뒤의 변위값을 예측할지 결정.
t_predict = 30
''' t_predict는 제작한 하드웨어에 설치된 가변저항 등을 이용해서,
실시간으로 조절하면서 사용할 수 있도록 하는 것도 좋겠다.
이름과 다르게 시간 스케일은 아니다. 예측 시간 간격은 t_predict*sampling_duration
'''


# 2. HF
# High Pass filter의 pass band를 결정하는 상수
RC = 1
''' 회로이론에서의 RC와 같은 의미
RC값이 작을 수록 강한 필터링. 즉 RC값이 무한이면 원래 신호가 나오고,
RC값이 0이면 모든 진동수의 성분이 제거되어 진폭이 0이 됨.
'''





#==========    Sample data    ==========
# noise를 포함한 sin파형
print("Enter Sampling Number : ")
n=int(input())

amp=3
ori_data = amp*np.sin(np.array(range(n))/100)
# sensor_noise 만큼의 표준편차를 가진 노이즈를 추가
noise_data = ori_data+np.random.normal(0,sensor_noise,n)





#========== Kalman filter, KF ==========
#https://ko.wikipedia.org/wiki/%EC%B9%BC%EB%A7%8C_%ED%95%84%ED%84%B0
kf_input = noise_data
# 예측값을 저장하는 행렬
kf_predict=[[],[],[]]
kf_result=[[],[],[]] 

# 각각의 필터는 입력과 출력만 잘 정의되면 독립적인 코드이므로, class로 구현하는 것도 좋을 것
# initial position, velocity, acceleration
pos=0
vel=-amp
'''offset설정을 위해 0이 아닌 값으로 초기화함.
안하면 평행이동한 그래프가 나오고, pos그래프가 증가하는 꼴의 그래프가 나온다.
그러나 DC필터에서 걸러지는 성분이므로, 어떤 값으로 초기화 시켜도 상관없다.'''
acc=0
x=np.array([[pos], [vel], [acc]])

# state transition matrix
A=np.array([[1,sampling_duration,sampling_duration**2/2],\
            [0,1,sampling_duration],[0,0,1]]) 

# P, (initial) covariance matrix of state vector
P = np.array([[100**4 * sensor_noise**2,0,0],\
              [0,100**2 * sensor_noise**2,0],[0,0,sensor_noise**2]])
''' 기본적으로 [[sigma_pos**2,0,0],[0,sigma_vel**2,0],[0,0,sigma_acc**2]]
으로 초기화 시키는 게 좋다. 위의 값은 sin파일 때만 적용 가능하며, 진동을 적절히 예측하고,
계산하여 초기화 시켜야 한다.
물론, 초기값이 정확하지 않더라도 충분한 시간이 지난 후에는 정확한 값으로 수렴하게 된다.
이 때, 초기의 오차 때문에 적분된 값이 평행이동 할 수 있는데, 이는 DC필터에서 제거되긴 하겠지만

'''
# R, (const) covariance of observation matrix
R = np.array([[sensor_noise**2]])
# Q, (const) covarience of process noise
B = np.array([[sampling_duration**3/6],[sampling_duration**2/2],[sampling_duration]])
Q = np.dot(B,np.transpose(B)) * sigma_jerk**2


# 사용자 입력은 없다.
H=np.array([[0,0,1]]) # 가속도만 입력변수로 받는다.

# 매 iteration마다 계산하는 것 방지
A_trans = np.transpose(A)
H_trans = np.transpose(H)


for i in range(n):
    z = np.array([[kf_input[i]]])
    
    x_ = np.dot(A,x) # + np.dot(B,u) 생략
    P_ = np.dot(A,np.dot(P,A_trans)) + Q
    
    # Kalman gain
    K = np.dot(np.dot(P_,H_trans),1/(np.dot(H,np.dot(P_,H_trans))+R))
    # 1차원 행렬이라 np.linalg.inv(np.dot(H,np.dot(P_,H_trans))+R) 역행렬 연산 필요없음

    x = x_ + np.dot(K,z-np.dot(H,x_))
    P = np.dot(np.array([[1,0,0],[0,1,0],[0,0,1]])-np.dot(K,H),P_)

    [kf_result[j].append(x[j][0]) for j in range(3)]
    
    ''' 결과확인을 위해 세 개의 결과값을 모두 저장했다.
    실제로 사용할 때는 위치를 나타내는 x[0][0]만 저장해도 충분하다.
    '''

    # kalman predict
    x_predict = x
    for j in range(t_predict):
        x_predict = np.dot(A,x_predict)
    [kf_predict[j].append(x_predict[j][0]) for j in range(3)]
 
    
# KF Result
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.title('Total '+str(n)+' time steps, '+'Predict '+str(t_predict)+' time steps using KF\n')
plt.plot([0,n],[0,0],color=(0,0,0))
plt.plot(kf_input, label='Input Data')
plt.plot(kf_result[2],color='r',label='Acc. (KF result)')
#plt.plot(kf_predict[2],color='g',label='Acc. (KF Predicted)')
# 가속도에 대해서는 어떤 예측도 안 한다. kf_result[2]와 kf_predict[2]는 같음
plt.ylabel("Acc.")
plt.grid(True)
plt.legend(fontsize='x-small')
plt.subplot(3,1,2)
plt.plot([0,n],[0,0],color=(0,0,0))
plt.plot(kf_input, label='Input Data')
plt.plot(kf_result[1],color='r',label='Vel. (KF result)')
plt.plot(kf_predict[1],color='g',label='Vel. (KF Predicted)')
plt.ylabel("Vel.")
plt.grid(True)
plt.legend(fontsize='x-small')
plt.subplot(3,1,3)
plt.plot([0,n],[0,0],color=(0,0,0))
plt.plot(kf_input, label='Input Data')
plt.plot(kf_result[0],color='r',label='Pos. (KF result)')
plt.plot(kf_predict[0],color='g',label='Pos. (KF Predicted)')
plt.ylabel("Pos.")
plt.xlabel("Time")
plt.grid(True)
plt.legend(fontsize='x-small')






#========== High pass Filter, HF ==========
#https://en.wikipedia.org/wiki/High-pass_filter
hf_input = kf_predict[0]
#hf_input = kf_result[0]

alpha = RC / (RC + sampling_duration)

hf_output = [hf_input[0]]
for i in range(1,n):
    hf_output.append(alpha*(hf_output[i-1]+hf_input[i]-hf_input[i-1]))


# HF result
plt.figure(figsize=(6,5))
plt.subplot(2,1,1)
plt.title('HPF with RC = '+str(RC)+'\n')
plt.plot([0,n],[0,0],color=(0,0,0))
plt.plot(kf_input)
plt.ylabel("Before HF")
plt.grid(True)
plt.plot(hf_input,color='r')

plt.subplot(2,1,2)
plt.plot([0,n],[0,0],color=(0,0,0))
plt.plot(kf_input)
plt.ylabel("After HF")
plt.xlabel("Time")
plt.grid(True)
plt.plot(hf_output,color='g')
plt.show()
