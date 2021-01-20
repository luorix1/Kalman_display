# KF, HPF filter class
import numpy as np

class Kalman:
    def __init__(self, A, H, R, Q, P, x, t_predict):
        '''
        A: 'const' state transition matrix
        H: 'const' observation matrix
        R: 'const' covariance of observation noise matrix 
        Q: 'const' covariance of process noise, function of sigma_jerk
        P: 'initial' covariance marix
        x: 'initial' state matrix
        '''
        self.A=A
        self.A_trans = A.transpose()
        self.H=H
        self.H_trans = H.transpose()
        self.R=R
        self.Q=Q
        self.P=P
        self.x=x
        self.t_predict=t_predict   
        
    def update(self, a, t_predict=None, Q=None):
        # 예측 시간간격과 sigma_jerk를 실시간으로 수정가능
        if t_predict: self.t_predict = t_predict
        if Q: self.Q = Q
            
        z = np.array([[a]])
        x_ = np.dot(self.A, self.x)
        P_ = np.dot(self.A, np.dot(self.P, self.A_trans)) + self.Q

        self.K = np.dot(P_, np.dot(self.H_trans, \
                1/(np.dot(self.H, np.dot(P_, self.H_trans)) + self.R)))
        self.x = x_ + np.dot(self.K, (z - np.dot(self.H, x_)))
        self.P = P_ - np.dot(self.K, np.dot(self.H, P_))
    
        x_predict = self.x
        for j in range(self.t_predict):
            x_predict = np.dot(self.A, x_predict)

        return (self.x[0][0], x_predict[0][0])
        # (filtered position, predicted position)


class HPF:
    def __init__(self, RC, sampling_duration):
        self.alpha = RC / (RC + sampling_duration)
        self.start=False
        
    def update(self, x):
        if self.start==False:
            self.start=True
            self.last_input = x
            self.last_output = x
            return x
        
        else:
            self.last_output = self.alpha * (self.last_output + x - self.last_input)
            self.last_input = x
            return self.last_output
        
