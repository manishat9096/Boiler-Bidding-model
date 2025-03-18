import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MPCController:
    def __init__(self, A, B, P, Q, horizon):
        self.A = A
        self.B = B
        self.P = P
        self.Q = Q
        self.horizon = horizon

    def simulate(self, u_init, y_init, yref):

        #def the variables
        u = cp.Variable(self.horizon)
        y = cp.Variable(self.horizon)

        constraints = [] # pass constraints as a list

        # initial conditions for the ARX Model
        # for i in range(0, len(y_init)):
        #     constraints += [y[i] == y_init[i]]
        # for i in range(0, len(u_init)):
        #     constraints += [u[i] == u_init[i]]
        for i in range(0, len(self.A)):
            constraints += [y[i] == y_init[i]]

        constraints += [u[0] == 0]
        constraints += [u[1] == 0]
        constraints += [u[2] == u_init[0]]
        constraints += [u[3] == u_init[1]]

        # rest of the prediction horizon
        for t in range(max(len(self.A), len(self.B)), self.horizon):
            # constraints += [y[t] == 
            #                 -sum(self.A[i] * y[t - i -1] for i in range(len(self.A))) +
            #                 sum(self.B[i] * u[t - i] for i in range(len(self.B)))
            #                 ]
            constraints += [y[t] == - self.A[0] * y[t-1] - self.A[1] * y[t-2] 
                                    - self.A[2] * y[t-3] - self.A[3] * y[t-4]
                                    + self.B[0] * u[t] + self.B[1] * u[t-1]]
            
        # boundary constraints
        constraints += [u>= 0, u<= 12] # MW limits
        constraints += [y >= 9, y<= 60] # Min 15% and max of the tank storage MWh

        # define the objective fn
        objective = 0
        for t in range(self.horizon):
            objective += self.Q * cp.norm(cp.square(y[t] - yref)) + self.P * cp.norm(cp.square(u[t]))
        # objective = cp.sum_squares(y - yref) * self.Q + cp.sum_squares(u) * self.P
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        if problem.status == 'Infeasible':
            print("Solver status:", problem.status)
        return u.value, y.value

if __name__ == '__main__':

    B = np.array([-0.00091497,  0.14029288])
    A = np.array([-1.04411331, 0.00942735,  0.00399007,  0.05331014])   
    horizon = 10
    P = 0.1
    Q = 1e3
    mpc = MPCController(A, B, P, Q, horizon)

    y_reference = 45
    u_init = np.zeros(2)
    y_init = np.ones(4) * 20
    u_opt, y_opt = mpc.simulate(u_init = u_init, y_init = y_init, yref= y_reference)
    print('Horizon (hours): ', horizon/4, ' P: ', P, ' Q: ', Q )
    if u_opt is not None:
        u_control =  [x for x in u_opt]
        y_output = [float("%.2f" % x) for x in y_opt]

        print("Optimal MW sequence:", [float("%.2f" % x) for x in u_opt])
        print("Predicted MWh sequence:", [float("%.2f" % x) for x in y_opt])

        time = range(len(y_output))
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Plot y_ref and y_output
        axs[0].axhline(y_reference, label = 'Y- reference')
        axs[0].plot(time, y_output, label="Predicted (y_output)", marker="o", color="blue")
        axs[0].set_ylabel("Storage Energy (MWh)")
        axs[0].legend()
        axs[0].grid(True)

        # Plot u_control
        axs[1].step(time, u_control, label="Control Input (u_control)", where="post", color="red")
        axs[1].set_xlabel("Time Step")
        axs[1].set_ylabel("Control Input (MW)")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()