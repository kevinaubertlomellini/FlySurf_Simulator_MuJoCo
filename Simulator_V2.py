import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from Generation_Automatic import *
from catenary_flysurf import *
from util import *
from LQR_functions import *

rows = 3 # Number of rows
cols = rows # Number of columns
x_init = -0.5
y_init = -0.5
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 1.75
shear_stif = 1.75
flex_stif = 2.25
g = 9.81
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1]]  # List of positions with special elements
#quad_positions = [[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, 1],[rows, cols]]  # List of positions with special elements
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols]]  # List of positions with special elements
quad_positions = [[1, 1],[rows, 1],[int((rows-1)/2)+1,int((cols-1)/2)+1],[1, cols],[rows, cols]]  #
mass_points = 0.001
m_total = (cols * rows)*mass_points
m = mass_points* np.ones((rows, cols))
mass_quads = 0.032
file_path = "config_FlySurf_Simulator.xml"  # Output file name

x_actuators = np.array(quad_positions)
n_actuators = x_actuators.shape[0]
print(n_actuators)

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

points_coord = x_actuators - 1
quad_indices = func_quad_indices(quad_positions, cols)

generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, file_path)
model = mujoco.MjModel.from_xml_path('scene_FlySurf_Simulator.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

time_change = 12
n_tasks = 2
total_time = time_change*n_tasks
time_step_num = round(total_time / model.opt.timestep)

time_num = 0

estimated_error = np.zeros([time_step_num,1])

n_points = rows
n_points2 = cols
l0= x_spacing
iter = time_step_num
c1 = 0.06
c2 = 0.06
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
x[2::6] = 0.5
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
    Q[6 * yu - 4, 6 * yu - 4] = 700 # Altitude
    Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] =  700 * np.eye(2) # x and y
    Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = 150 * np.eye(2)  # velocity

R = 250 * np.eye(3 * n_actuators) # force in z
for yi in range(n_actuators):
    R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = 150 * np.eye(2) # force in x and y

K = k_dlqr(n_points,n_points2,str_stif,shear_stif,flex_stif,c1,c2,l0,m,mass_quads,x_actuators,x,Q,R,n_visible_points,x_visible_points_2,model.opt.timestep)

u_lim_T = 5.0* np.array([-1.0, 1.0])
u_lim_M = 5.0 * np.array([-1.0, 1.0])
u_Forces = np.zeros((3*n_actuators,1))
M_inv = np.linalg.inv(M)

for kv in range(1, n_actuators + 1):
    forces = np.array([0, 0, (m_total + (mass_quads-mass_points)*n_actuators)*g / n_actuators])
    u_Forces[3 * kv - 3:3 * kv, 0] = forces.flatten()

import numpy as np

# Constants
alpha_H = 2
alpha_G = 20
alpha_0 = 2
alpha_Hd = 3

# Reshape shape
shape = np.reshape(np.array([xd[0::6], xd[1::6], xd[2::6]]).T, (-1, 1)).flatten()

# Define rotation matrix (assuming rotationMatrix(0, 0, 0) is identity)
R_d = np.eye(3)
s_d = 1
c_0 = np.array([0, 0, 0])
factor=0.125

# Call x_des (assuming the function is defined)
x_c = x_des(n_points, n_points2, shape, R_d, s_d, c_0)

# Generate mesh grid for x and y coordinates
x_g = np.linspace(-0.375, 0.375, n_points)
y_g = np.linspace(-0.375, 0.375, n_points2)
X_g, Y_g = np.meshgrid(x_g, y_g)

x_g_vector0 = X_g.flatten()
y_g_vector0 = Y_g.flatten()

# Define Gaussian parameters
Amp = 1.1  # Amplitude
x0 = 0.0  # Center of Gaussian in x
y0 = 0.0  # Center of Gaussian in y
sigma_x = 0.575  # Standard deviation in x
sigma_y = 0.575  # Standard deviation in y

# Calculate the 2D Gaussian
z_g_vector0 = Amp * np.exp(-((x_g_vector0 - x0) ** 2 / (2 * sigma_x ** 2) +
                             (y_g_vector0 - y0) ** 2 / (2 * sigma_y ** 2))) - 0.5

# Combine into shape_gaussian
shape_gaussian = np.vstack((x_g_vector0, y_g_vector0, z_g_vector0)).T.flatten()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()

        print(xd)

        # Reshape xd to extract positions
        xd_pos = np.column_stack((xd[::6], xd[1::6], xd[2::6]))

        print('xd_pos',xd_pos)

        # Call the shape_controller_3D function
        u_shape = shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, xd_pos, n_points * n_points2, shape, R_d,
                                      s_d, c_0)

        # Update xd_pos with the computed control input
        xd_pos = xd_pos + u_shape * factor * model.opt.timestep

        # Combine updated positions and control input
        xd = np.reshape(
            np.vstack((np.reshape(xd_pos, (3, -1)), np.reshape(factor * u_shape, (3, -1)))).T.flatten(),
            (-1, 1)
        ).flatten()

        print(xd)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points with black stars and a linewidth of 1.0
        ax.plot3D(xd[0::6], xd[1::6], xd[2::6], 'k*', linewidth=1.0)

        # Show the plot
        plt.show()

        states = data.xpos
        vels = data.cvel
        combined = np.hstack((states[1:], vels[1:,:3]))
        x = combined.flatten().reshape(-1, 1)

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

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        time_num += 1
        if time_num >= time_step_num:
            break

        if 1*time_change< time_num*model.opt.timestep < 1.6*time_change:
            xd[2::6] = xd[2::6] + 0.75 / (time_change/model.opt.timestep)
            xd[::6] = xd[::6] + 0.75 /  (time_change/model.opt.timestep)

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
