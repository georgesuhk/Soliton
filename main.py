
#Runge Kutta 2nd Order
import numpy as np
#import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from simulations import *
from post import *

anim_folder = "animations/"



#%%
time_step = 0.01
rk_alpha = 0.5
soliton_alpha = 0.5
h = 2.1

X = np.linspace(-50,100,200)
t_lims = [0,50]

U_0 = U_init(X, soliton_alpha, 1, t_lims[0])
soliton_1_amp = 5
soliton_1_0 = U_init(X, soliton_alpha, soliton_1_amp, t_lims[0])


soliton_1 = RK4(t_lims, U_0, time_step, rk_alpha, h)
time_list = soliton_1[0]
Un_list = soliton_1[1]
Un_snapshot = soliton_1[2]


soliton_2 = RK4(t_lims, soliton_1_0, time_step, rk_alpha, soliton_1_amp*h)
time_list_2 = soliton_2[0]
Un_list_2 = soliton_2[1]



quick_plot(X, soliton_1)


#%% Animate

    

gen_anim_1(X, soliton_1, anim_folder)

