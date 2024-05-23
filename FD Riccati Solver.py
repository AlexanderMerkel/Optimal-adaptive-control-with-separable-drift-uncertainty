import numpy as np
from scipy.integrate import odeint
import torch as th

y_min = -3; y_max = 3; m_min = -5.0; m_max = 5.0; g_min = 0.0; g_max = 10.0
y0 = 0; T = 1; sig = 1; c = 2; rho = 2; C = 5

def riccati(y, t, a, c):
    return - a * y**2 + c

def solve_riccati_odeint(a, c, y0, T, dt):
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)
    y = odeint(riccati, y0, t, args=(a, c))
    return t, y

yT = C
dt = 1.0e-4
dlambda = 1.0e-4
lam_range = np.arange(0, 1, dlambda)

import matplotlib.pyplot as plt

Solution = th.zeros((len(lam_range), int(T/dt)+1))
for lam in lam_range:
    a = lam ** 2 / (rho **2)
    t, y = solve_riccati_odeint(a, c, yT, T, dt)
    
    t_tensor = th.from_numpy(t).float().squeeze()
    y_tensor = th.from_numpy(y).float().squeeze().__reversed__()
    
    Solution[int(lam*(1 / dlambda)), :] = y_tensor
    plt.plot(t_tensor.numpy(), y_tensor.numpy(), label=f'a={a}')
print(Solution.shape)
th.save(Solution, 'outputs/FD_Riccati.pt')
plt.show()