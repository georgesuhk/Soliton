
import numpy as np
#import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from simulations import *
from post import *

anim_folder = "animations/"



#%% Simulate


rk_alpha = 3.0
soliton_alpha = 1
initial_snapshot = True

X = np.linspace(-20,20,200)
h = np.diff(X)[0]
dt = h**3
n_steps = 3000


#analytic soliton
fast_soliton_0 = analytic(X, soliton_alpha*1.2, -2)
slow_soliton_0 = analytic(X, soliton_alpha, -1.5)

double_soliton_0 = fast_soliton_0 + slow_soliton_0


if initial_snapshot:
    plt.figure()
    plt.plot(X, fast_soliton_0, label='fast')
    plt.plot(X, slow_soliton_0, label= 'slow')
    #plt.plot(X, double_soliton_0, label='both')
    plt.title("initial snapshot")
    plt.show()



#%% Animating
#propagating for collision
double_soliton = RK4_vec(double_soliton_0, n_steps = n_steps, dt = dt, h = h)
quick_plot(X, double_soliton, 'double soliton')

gen_anim_1(X, double_soliton, anim_folder, "collision")






