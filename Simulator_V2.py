import mujoco.viewer
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

from Generation_Automatic import *
from catenary_flysurf import *
from util import *
from LQR_MPC_functions import *

# SHAPE PLANNER ADDED

rows = 19# Number of rows
cols = rows # Number of columns
x_init = -0.5
y_init = -0.5
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.25
shear_stif = 0.005
flex_stif = 0.005
g = 9.81
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1]]  # List of positions with special elements
#quad_positions = [[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, 1],[rows, cols]]  # List of positions with special elements
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols]]  # List of positions with special elements
quad_positions = [[1, 1],[rows, 1],[int((rows-1)/2)+1,int((cols-1)/2)+1],[1, cols],[rows, cols]]  #
#quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]
mass_points = 0.002
m_total = (cols * rows)*mass_points
m = mass_points* np.ones((rows, cols))
mass_quads = 0.032
damp_point = 0.01
damp_quad = 0.06
file_path = "config_FlySurf_Simulator.xml"  # Output file name

x_actuators = np.array(quad_positions)
n_actuators = x_actuators.shape[0]
print(n_actuators)

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

points_coord = x_actuators - 1
quad_indices = func_quad_indices(quad_positions, cols)

generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, file_path)

# Set up the video writer for .mp4 format
frame_width = 640  # Set the width of the video
frame_height = 480  # Set the height of the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use the 'mp4v' codec for .mp4 format
out = cv2.VideoWriter('simulation_output.mp4', fourcc, 30, (frame_width, frame_height))  # 30 FPS

model = mujoco.MjModel.from_xml_path('scene_FlySurf_Simulator.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

time_change = 12
n_tasks = 6
total_time = time_change*n_tasks
time_step_num = round(total_time / model.opt.timestep)

time_num = 0

estimated_error = np.zeros([time_step_num,1])

n_points = rows
n_points2 = cols
l0= x_spacing
iter = time_step_num
c1 = damp_point
c2 = damp_quad
n_visible_points = n_actuators
x_actuators_2 = np.zeros((n_actuators, 3))
for i in range(n_actuators):
    m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = mass_quads
    x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 0
    x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
    x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
x_visible_points_2 = x_actuators_2
u_save = np.zeros((3*n_actuators, iter))
x_save = np.zeros((6*n_points*n_points2, iter))
e_save = np.zeros((iter,1))
e_save2 = np.zeros((iter,1))
t_save = np.zeros((iter, 1))
xd_save = np.zeros((6*n_points*n_points2, iter))
x = np.zeros((n_points * n_points2 * 6,1))

for i in range(n_points):
    for j in range(n_points2):
        x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
        x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)

x[0::6] = x[0::6] - 0.5
x[1::6] = x[1::6] - 0.5
x[2::6] = 0
xd = np.copy(x)

M = np.eye(6 * n_points * n_points2)
G = np.zeros((6 * n_points * n_points2,1))
# Update M and G
for j in range(n_points2):
    for i in range(n_points):
        pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
        M[pos_v, pos_v] = m[i, j] * np.eye(3)
        G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
# print("M:", M)
# print("G:", G)

Q = 1*np.eye(6 * n_points * n_points2)
for yu in range(1, n_points * n_points2 + 1):
    Q[6 * yu - 4, 6 * yu - 4] = 370 # Altitude
    Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] =  450 * np.eye(2) # x and y
    Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = 150 * np.eye(2)  # velocity

R = 260 * np.eye(3 * n_actuators) # force in z
for yi in range(n_actuators):
    R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = 200 * np.eye(2) # force in x and y

print(model.opt.timestep)

K = k_dlqr(n_points,n_points2,str_stif,shear_stif,flex_stif,c1,c2,l0,m,mass_quads,x_actuators,x,Q,R,n_visible_points,x_visible_points_2,model.opt.timestep)

u_lim_T = 5.0* np.array([-1.0, 1.0])
u_lim_M = 5.0 * np.array([-1.0, 1.0])
u_Forces = np.zeros((3*n_actuators,1))
M_inv = np.linalg.inv(M)

for kv in range(1, n_actuators + 1):
    forces = np.array([0, 0, (m_total + (mass_quads-mass_points)*n_actuators)*g / n_actuators])
    u_Forces[3 * kv - 3:3 * kv, 0] = forces.flatten()

print(u_Forces)

# Define parameters
alpha_H = 2
alpha_G = 20
alpha_0 = 2
alpha_Hd = 3

shape = np.reshape(np.array([x[::6],x[1::6],x[2::6]]),(3, n_points * n_points2)).reshape(-1,1, order='F')


R_d = rotation_matrix(0, 0, 0)
s_d = 1
c_0 = np.array([0, 0, 0.5])

# Generate mesh grid for x and y coordinates
x_g = np.linspace(-0.375, 0.375, n_points)
y_g = np.linspace(-0.375, 0.375, n_points2)
X_g, Y_g = np.meshgrid(x_g, y_g)

x_g_vector0 = X_g.flatten()
y_g_vector0 = Y_g.flatten()

# Define Gaussian parameters
Amp = 1.12  # Amplitude
x0 = 0.0  # Center of Gaussian in x
y0 = 0.0  # Center of Gaussian in y
sigma_x = 0.575  # Standard deviation in x
sigma_y = 0.575  # Standard deviation in y
factor=0.15

# Calculate the 2D Gaussian
z_g_vector0 = Amp * np.exp(-((x_g_vector0 - x0) ** 2 / (2 * sigma_x ** 2) +
                             (y_g_vector0 - y0) ** 2 / (2 * sigma_y ** 2))) - 0.5

# Combine into shape_gaussian
shape_gaussian = np.vstack((x_g_vector0, y_g_vector0, z_g_vector0)).T.flatten()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()
        if time_num == 0:
            viewer.cam.lookat[0] = 0.5  # Move camera target in the x-direction
            viewer.cam.lookat[1] = -0.65  # Move camera target in the y-direction
            viewer.cam.lookat[2] = 1  # Move camera target in the z-direction
            viewer.cam.distance = 2.0  # Zoom out
            viewer.cam.azimuth = 90  # Change azimuth angle
            viewer.cam.elevation = -30  # Change elevation angle

        states = data.xpos
        vels = data.cvel
        combined = np.hstack((states[1:], vels[1:,:3]))
        x = combined.flatten().reshape(-1, 1)

        xd_pos = np.reshape(np.array([xd[::6], xd[1::6], xd[2::6]]), (3, n_points * n_points2)).reshape(-1, 1, order='F')
        u_shape = shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, xd_pos, n_points * n_points2, shape, R_d, s_d, c_0)
        xd_pos = xd_pos + u_shape * factor * model.opt.timestep

        combined2 = np.hstack((np.reshape(xd_pos, (n_points*n_points2, -1)), np.reshape(factor * u_shape, (n_points*n_points2, -1))))
        xd = combined2.flatten().reshape(-1, 1)

        u2 = np.dot(K, (xd - x))

        u = u2 + u_Forces  # Compute control inputs for all drones

        for kv in range(1, n_actuators + 1):
            # Check and update u[3*kv-1]
            if u[3 * kv - 3] < u_lim_M[0]:
                u[3 * kv - 3] = u_lim_M[0]
            elif u[3 * kv - 3] > u_lim_M[1]:
                u[3 * kv - 3] = u_lim_M[1]

            # Check and update u[3*kv]
            if u[3 * kv - 2] < u_lim_M[0]:
                u[3 * kv - 2] = u_lim_M[0]
            elif u[3 * kv - 2] > u_lim_M[1]:
                u[3 * kv - 2] = u_lim_M[1]

            # Check and update u0[3*kv-2]
            if u[3 * kv - 1] < u_lim_T[0]:
                u[3 * kv - 1] = u_lim_T[0]
            elif u[3 * kv - 1] > u_lim_T[1]:
                u[3 * kv - 1] = u_lim_T[1]

        data.ctrl[:] = u.flatten()

        mujoco.mj_step(model, data)


        u_save[:, time_num] = u2.flatten()

        x_save[:, time_num] = x.flatten()
        xd_save[:, time_num] = xd.flatten()

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Render the simulation scene
        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        time_num += 1
        if time_num >= time_step_num:
            break
        if time_change == 1.0 * time_num*model.opt.timestep:
            shape = shape_gaussian
        if 2.0*time_change == time_num*model.opt.timestep:
            R_d = rotation_matrix(0, -np.pi / 4, np.pi / 2)
        if (3.0 * time_change < time_num*model.opt.timestep) & (5.0 * time_change > time_num*model.opt.timestep):
            c_0_x = np.cos((time_num - 3.0 * (time_change/model.opt.timestep)) /(time_change/model.opt.timestep) * np.pi) - 1
            c_0_y = np.sin((time_num- 3.0 * (time_change/model.opt.timestep)) /(time_change/model.opt.timestep) * np.pi)
            c_0 = 0.5 * np.array([c_0_x, c_0_y, 1.0])
            factor = 3
        '''
        if 1*time_change == time_num*model.opt.timestep:
            x1 = (x_g_vector0.reshape(-1, 1) - xd[::6])/(1.5*time_change/model.opt.timestep)
            y1 = (y_g_vector0.reshape(-1, 1) - xd[1::6])/(1.5*time_change/model.opt.timestep)
            z1 = (z_g_vector0.reshape(-1, 1) - xd[2::6])/(1.5*time_change/model.opt.timestep)
        if 1*time_change< time_num*model.opt.timestep < 1.5*time_change:
            xd[::6] = xd[::6] + x1
            xd[1::6] = xd[1::6] + y1
            xd[2::6] = xd[2::6] + z1
        if 2*time_change< time_num*model.opt.timestep < 2.6*time_change:
            xd[2::6] = xd[2::6] + 0.75 / (time_change/model.opt.timestep)
            xd[::6] = xd[::6] + 0.75 /  (time_change/model.opt.timestep)
        
        if time_num ==1:
            for i in range(7):
                time.sleep(1.0)
                print(i)
        '''
    # Release the video writer once done
    out.release()

t = np.arange(0, iter * model.opt.timestep, model.opt.timestep)
cmap = plt.get_cmap("tab10")

plt.figure(1)
plt.plot(t,x_save[0, :], label='xe', color=cmap(0), linewidth=1.5)
plt.plot(t,xd_save[0, :], '--', label='x_d', color=cmap(0), linewidth=1.5, alpha=0.3)
plt.plot(t,x_save[1, :], label='ye', color=cmap(1), linewidth=1.5)
plt.plot(t,xd_save[1, :], '--', label='y_d', color=cmap(1), linewidth=1.5, alpha=0.6)
plt.plot(t,x_save[2, :], label='ze', color=cmap(2), linewidth=1.5)
plt.plot(t,xd_save[2, :], '--', label='z_d', color=cmap(2), linewidth=1.5, alpha=0.6)
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Position CF1")
plt.legend()
plt.grid(True)

# Plot 1: Control signal u_z
plt.figure(2)
plt.plot(t, u_save[2, :], linewidth=1.5)
plt.plot(t, u_save[5, :], linewidth=1.5)
plt.ylabel("Force (N)")
plt.xlabel("Time (s)")
plt.title("Force Fz")
plt.grid(True)

for i in range(iter):
    pos = np.array([x_save[::6, i], x_save[1::6, i], x_save[2::6, i]]).T
    pos_d = np.array([xd_save[::6, i], xd_save[1::6, i], xd_save[2::6, i]]).T
    np.mean(np.linalg.norm(pos - pos_d, axis=1))
    e_save[i] = average_hausdorff_distance(pos, pos_d)
    e_save2[i] = np.mean(np.linalg.norm(pos - pos_d, axis=1))

plt.figure(9)
plt.plot(t, e_save.flatten(), linewidth=1.5, label='AHD')
plt.plot(t, e_save2.flatten(), linewidth=1.5, label='ED')
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.title("Error")
plt.legend()
plt.grid(True)
plt.show()
