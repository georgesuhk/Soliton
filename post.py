# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def quick_plot(X, sim, title):
    Un_list = sim[1]
    plt.figure()
    
    plt.plot(X,Un_list[0],label="start",color="#432371")
    plt.plot(X,Un_list[round(len(Un_list)/4)], color="#714674")
    plt.plot(X,Un_list[round(len(Un_list)*1/2)], color="#9F6976")
    plt.plot(X,Un_list[round(len(Un_list)*3/4)], color="#CC8B79")
    plt.plot(X,Un_list[-1], label="end", color = "#FAAE7B")
    plt.xlim(min(X),max(X))
    plt.xlabel("x")
    plt.ylabel("U")
    plt.title(title)
    #plt.title('alpha = %.2f, time_step = %.5f, h = %.2f' % (1, time_step, h))
    
    plt.legend()
    plt.show()
    

#generate animation for 1 soliton
def gen_anim_1(X, sim, output_folder, output_name, plot_interval):
    
    time_list = sim[0]
    Un_list = sim[1]
    U_0 = Un_list[0]
    
    num_frames = round(len(time_list)/plot_interval)
    
    def animate(i):
         
        U = Un_list[i*plot_interval]
        ax.clear()
        ax.plot(X, U)
        ax.set_xlim([min(X),max(X)])
        ax.set_ylim([min(U_0)*0.75, 2*max(U_0)])
    
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=200, repeat=False)
    writergif = animation.PillowWriter(fps=100)
    anim.save(output_folder+output_name+".gif", writer=writergif)
    
    
#generate animation for 2 soliton #outdated
def gen_anim_2(X, sim1, sim2, output_folder, output_name):
    
    
    
    time_list = sim1[0]
    Un_list_1 = sim1[1]
    Un_list_2 = sim2[1]

    U_init_1 = Un_list_1[0]
    U_init_2 = Un_list_2[0]
    
    def animate(i):
         
        U_1 = Un_list_1[i]
        U_2 = Un_list_2[i]
        
        ax.clear()
        ax.plot(X, U_1)
        ax.plot(X, U_2/sim2_amp)

        ax.set_xlim([-20,max(X)])
        ax.set_ylim([min(U_init_1)*0.75, 2*max(U_init_2)])
        ax.set_title("Co-plotted solitons (different simulations)")
        ax.set_xlabel("x")
        ax.set_ylabel("U")
    
    fig, ax = plt.subplots()
    anim = animation.FuncAnimation(fig, animate, frames=len(time_list), interval=200, repeat=False)
    writergif = animation.PillowWriter(fps=100)
    anim.save(output_folder+output_name+".gif", writer=writergif)