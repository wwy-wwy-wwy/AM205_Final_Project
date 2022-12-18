#steps:
#1. set the reference trajectory 
#2. define the state and control boundaries
#3. initialize the predictive states p 
#4. initialize the control states u 
#5. update the state states in each time step(mpc control update)
#calculate the optimization solver
#the control vectore generated from the solver is then used to minimize the cost function
#7. calculate the cost function that finds the error of the predicted states and control states and minimize it
#8. the cost function is needed for the control update 
#9. find the optimal path according to the resulting state states and a reference states (polynomial fit)

import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.optimize import minimize






def predict_model(prev_state, dt, throt, steering):
         

        x_curr = prev_state[0] + np.cos(prev_state[2]) * prev_state[3] * dt
        y_curr = prev_state[1] + np.sin(prev_state[2]) * prev_state[3] * dt

        a_curr = throt
        v_curr = prev_state[3] + a_curr * dt - prev_state[3]/25

        psi_curr = prev_state[2] + v_curr * dt * np.tan(steering)/Lf

        return [x_curr, y_curr, psi_curr, v_curr]


def cost_function(u, *args):
        p_state = args[0]
        ref_state = args[1]
        cost = 0

        for i in range(control_horizon):
            velocity = p_state[3]
            psi = p_state[2]

            p_state = predict_model(p_state,dt, u[i*2], u[i*2 + 1])

            ref_dist = np.sqrt((ref_state[0] - p_state[0])**2 + (ref_state[1] - p_state[1])**2)
            obstacle_dist = np.sqrt((x_obs - p_state[0])**2 + (y_obs - p_state[1])**2)

            # Position error cost
            cost +=  ref_dist #this cost doen't have high weight

            # Obstacle distance cost
            if obstacle_dist < 1.5:
                cost += 3.5/obstacle_dist # the value 0.1 represents the weight of the cost

            # psi error cost
            cost += 10 * (psi - p_state[2])**2 #weight is 10

            cost +=  2 * (ref_state[2] - p_state[2])**2 #weight is 2

            # throttle error cost
            if abs(u[2*i]) > 0.2:
                cost += (velocity - p_state[3])**2

        return cost

#define some variables for the state update
dt=0.2
Lf = 2.5
control_horizon = 15
#set the reference trajectory
ref_traj = pd.read_csv("lake_track_waypoints.csv")
nt = len(ref_traj['x']) #no. of time step
x_obs = 7
y_obs = 0.1
#define state variables (actual trajectory)
#p_traj = np.zeros(nt, 4)
#define states variables' boundaries
v_bound = (0, 128*pi/180)
#initialize the states [x, y, psi, v]
p_init = np.array([[0,0,0,0]])
state_pred = [p_init]
#define control variables
num_control = 2
u_traj = np.zeros(control_horizon*num_control)
#define control variables' boundaries
u_bounds = []
for i in range(control_horizon):
        u_bounds += [[-1, 1]]
        u_bounds += [[-0.4, 0.4]] #25 degrees in radian

#initialize the control variables
u_init = np.array([[0,0]])
ref_point = [ref_traj['x'][0],ref_traj['y'][0],0]
xval = ref_traj['x'][0]
yval = ref_traj['y'][0]
print('nt issss'+str(nt))
for i in range(1,nt+1):
    #shift the first two control parameters and add the newly added two parameters
    u_traj = np.delete(u_traj,0)
    u_traj = np.delete(u_traj,0)
    u_traj = np.append(u_traj, u_traj[-2])
    u_traj = np.append(u_traj, u_traj[-2])

    ###step1:minimize the cost function to optimize the control parametrs by doing nonlinear optimization###
    ref_point = [10, 0,0]
    #ref_point = [ref_traj['x'][i],ref_traj['y'][i],0]
    #print('ref traj is'+str(ref_point))
    #ref_point = [10, 0, 0]
    u_opti = minimize(cost_function, u_traj, (p_init[-1], ref_point),
                            method='SLSQP',
                            bounds=u_bounds,
                            tol = 1e-5)
    u_traj = u_opti.x
    #####step2: predict the next state variables using the first control pair  ###
    res = predict_model(p_init[-1], dt, u_traj[0], u_traj[1])

    p_pred = np.array([res])
    #####step3: predict the rest of the state variables using the rest of the control pairs ###
    for j in range(1, control_horizon):
        predicted =predict_model(p_pred[-1], dt, u_traj[2*j], u_traj[2*j+1])
        p_pred = np.append(p_pred, np.array([predicted]), axis=0)
    state_pred += [p_pred]
    p_init = np.append(p_init, np.array([res]), axis=0)
    u_init = np.append(u_init, np.array([(u_traj[0], u_traj[1])]), axis=0)

 



###################
# SIMULATOR DISPLAY

# Total Figure
fig = plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(8,8)

# Elevator plot settings.
ax = fig.add_subplot(gs[:8, :8])

ax.set_xlim(-3, 17)
ax.set_ylim([-3, 17])
plt.xticks(np.arange(-10,10, step=50))
plt.yticks(np.arange(-10,10, step=50))
plt.title('MPC 2D')

# Time display.
time_text = ax.text(6, 0.5, '', fontsize=15)

# Main plot info.
car_width = 1.0
patch_car = mpatches.Rectangle((0, 0), car_width, 2.5, fc='k', fill=False)
#patch_goal = mpatches.Rectangle((0, 0), car_width, 2.5, fc='b', #ls='dashdot', fill=False)

ax.add_patch(patch_car)
#ax.add_patch(patch_goal)
predict, = ax.plot([], [], 'r--', linewidth = 1)

# Car steering and throttle position.
telem = [3,14]

######
#patch_wheel = mpatches.Circle((telem[0]-3, telem[1]), 2.2)
#ax.add_patch(patch_wheel)
######

#wheel_1, = ax.plot([], [], 'k', linewidth = 3)
#wheel_2, = ax.plot([], [], 'k', linewidth = 3)
#wheel_3, = ax.plot([], [], 'k', linewidth = 3)

'''
throttle_outline, = ax.plot([telem[0], telem[0]], [telem[1]-2, telem[1]+2],'b', linewidth = 20, alpha = 0.4)
throttle, = ax.plot([], [], 'k', linewidth = 20)
brake_outline, = ax.plot([telem[0]+3, telem[0]+3], [telem[1]-2, telem[1]+2],'b', linewidth = 20, alpha = 0.2)
brake, = ax.plot([], [], 'k', linewidth = 20)
throttle_text = ax.text(telem[0], telem[1]-3, 'Forward', fontsize = 15,horizontalalignment='center')
brake_text = ax.text(telem[0]+3, telem[1]-3, 'Reverse', fontsize = 15,horizontalalignment='center')
'''
# Obstacles

patch_obs = mpatches.Circle((x_obs, y_obs),0.5)
ax.add_patch(patch_obs)

# Shift xy, centered on rear of car to rear left corner of car.
def car_patch_pos(x, y, psi):
    #return [x,y]
    x_new = x - np.sin(psi)*(car_width/2)
    y_new = y + np.cos(psi)*(car_width/2)
    return [x_new, y_new]
 
def update_plot(num):
    # Car.
    patch_car.set_xy(car_patch_pos(p_init[num,0], p_init[num,1], p_init[num,2]))
    patch_car.angle = np.rad2deg(p_init[num,2])-90
    # Car wheels
    np.rad2deg(p_init[num,2])
    #steering_wheel(u_i[num,1]*2)
    #throttle.set_data([telem[0],telem[0]],
                    #[telem[1]-2, telem[1]-2+max(0,u_i[num,0]/5*4)])
    #brake.set_data([telem[0]+3, telem[0]+3],
                    #[telem[1]-2, telem[1]-2+max(0,-u_i[num,0]/5*4)])

    # Goal.
    #if (num <= 130 or ref_2 == None):
        #patch_goal.set_xy(car_patch_pos(ref_1[0],ref_1[1],ref_1[2]))
        #patch_goal.angle = np.rad2deg(ref_1[2])-90
    #else:
        #patch_goal.set_xy(car_patch_pos(ref_2[0],ref_2[1],ref_2[2]))
        #patch_goal.angle = np.rad2deg(ref_2[2])-90

    #print(str(state_i[num,3]))
    predict.set_data(state_pred[num][:,0],state_pred[num][:,1])
    # Timer.
    #time_text.set_text(str(100-t[num]))

    return patch_car, time_text


#print("Compute Time: ", round(time.process_time() - start, 3), "seconds.")
# Animation.
car_ani = animation.FuncAnimation(fig, update_plot, frames=range(1,len(p_init)), interval=100, repeat=True, blit=False)
car_ani.save('mpc-video.mp4')

plt.show()
















