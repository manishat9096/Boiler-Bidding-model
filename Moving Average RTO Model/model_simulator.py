import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ARXSimulator:
    def __init__(self, A, B):
        self.A = np.array(A)
        self.B = np.array(B)
        self.y_lags = len(A)
        self.u_lags = len(B)

    def ARX_model(self):

        y_term = -sum(self.A[i] * self.y_past_sim[-(i+1)] for i in range(self.y_lags))
        u_term = sum(self.B[i] * self.u_past_sim[-(i+1)] for i in range(self.u_lags))
        return y_term + u_term

    def MW_to_MWh(self, input_sequence, y_init, u_init):
        self.y_past_sim = np.array(y_init, dtype=float)
        self.u_past_sim = np.array(u_init, dtype=float) 
        
        input_sequence = np.array(input_sequence, dtype=float)
        y_predict = np.zeros(len(input_sequence))

        for i in range(len(input_sequence)):
            y_next = self.ARX_model()
            y_predict[i] = y_next
            self.y_past_sim = np.append(self.y_past_sim, y_next)[-self.y_lags:]
            self.u_past_sim = np.append(self.u_past_sim, input_sequence[i])[-self.u_lags:]

        return y_predict

if __name__ == '__main__':

    A = [-1.04411331, 0.00942735,  0.00399007,  0.05331014] 
    B = [-0.00091497,  0.14029288]  
    y_init = [20, 20, 20, 20]  
    u_init = [0, 0]

    ARX = ARXSimulator(A, B, y_init, u_init)

    # Test with a decreasing MW input sequence
    print(ARX.MW_to_MWh([6, 6, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0,0,0,0,0,0,0,0,0,0]))

    for ip in (np.ones(5) * 6):
        print(ARX.MW_to_MWh([ip]))