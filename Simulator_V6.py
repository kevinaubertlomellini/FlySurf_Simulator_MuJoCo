import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time

from Generation_Automatic import *
from catenary_flysurf import *
from util import *
from LQR_MPC_functions import *

# FLYSURF SIMULATOR PARAMETERS
rows = 21# Number of rows
cols = rows # Number of columns
x_init = -0.5 # Position of point in x (1,1)
y_init = -0.5 # Position of point in y (1,1)
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.025 # Stifness of structural springs
shear_stif = 0.005 # Stifness of shear springs
flex_stif = 0.005 # Stifness of flexion springs
g = 9.81 # Gravity value
quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols],[3,int((cols-1)/2)+1]]  # UAVs positions in the grid simulator
#quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]
mass_points = 0.0025 # Mass of each point0
mass_quads = 0.32 # Mass of each UAV
damp_point = 0.001 # Damping coefficient on each point
damp_quad = 0.6 # Damping coefficient on each UAV
T_s = 0.002 # Simulator step
u_limits = 100*np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]) # Actuator limits
file_path = "FlySurf_Simulator.xml"  # Output xml file name

# Generate xml simulation  file
[model, data] = generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, T_s, u_limits, file_path)

[x_actuators, n_actuators] = init_simulator(quad_positions)
print(x_actuators)

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

delta_factor = 5
delta = delta_factor*T_s
time_change = 20
n_tasks = 3
total_time = time_change*n_tasks
time_step_num = round(total_time / T_s)


n_points = int((rows+1)/2)
n_points2 = int((cols+1)/2)
l0= 2*x_spacing
iter = int(time_step_num/delta_factor)

[u_save, x_save, xd_save, xe_save] = init_vectors(n_actuators, [n_points, n_points2], iter)

x = np.zeros((n_points * n_points2 * 6,1))
for i in range(n_points):
    for j in range(n_points2):
        x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
        x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)
x[0::6] = x[0::6] - 0.5
x[1::6] = x[1::6] - 0.5
x[2::6] = 0
xd = np.copy(x)

# CONTROL PARAMETERS
Q_vector = [500, 1500, 150, 200, 1] # [x and y, z, velocity in x and y, velocity in z]
R_vector = [200, 320] # [force in x and y, force in z]
K = k_dlqr_V2(n_points,n_points2,str_stif,shear_stif,flex_stif,damp_point,damp_quad,l0,mass_points,mass_quads,x_actuators,x,Q_vector,R_vector,delta)
u_gravity = u_gravity_forces(n_UAVs = n_actuators, mass_points = mass_points, mass_UAVs = mass_quads, rows =rows, cols=cols, g= g)
print(u_gravity)

# PATH PLANNING PARAMETERS
alpha_H = 3
alpha_G = 20
alpha_0 = 2
alpha_Hd = 3
shape = np.reshape(np.array([x[::6],x[1::6],x[2::6]]),(3, n_points * n_points2)).reshape(-1,1, order='F')
R_d = rotation_matrix(0, 0, 0)
s_d = 1
c_0 = np.array([0, 0, 0.7])
factor=0.5
shape_gaussian = shape_gaussian_mesh(sides=[0.8, 0.8], amplitude=1.12, center=[0.0, 0.0], sd = [0.575, 0.575], n_points = [n_points, n_points2])

indices = []
for i in range(1,n_points+1):
    for j in range(1,n_points2+1):
        indices.append(int((2*i-2)*rows + (2*j-1)))

flysurf = CatenaryFlySurf(n_points2, n_points, l0+0.0011, num_sample_per_curve=n_points2)

quad_positions2 = [[rows, cols], [rows, 1], [1, 1], [1, cols],
                   [int((rows - 1) / 2) + 1, int((cols - 1) / 2) + 1],[3,int((cols+1)/2)]]

[points_coord2, quad_indices2] = points_coord_estimator(quad_positions2, rows, cols)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

start_time = time.time()  # Record start time
time_num = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat = [0.5, -0.65, 1]  # Move camera target in the [x, y, z] direction
    viewer.cam.distance = 2.0  # Zoom out
    viewer.cam.azimuth = 90  # Change azimuth angle
    viewer.cam.elevation = -30  # Change elevation angle
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0

    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()

        if time_num%delta_factor ==0:
            states = data.xpos
            vels = data.cvel
            combined = np.hstack((states[indices], vels[indices,:3]))
            x = combined.flatten().reshape(-1, 1)

            xd_pos = np.reshape(np.array([xd[::6], xd[1::6], xd[2::6]]), (3, n_points * n_points2)).reshape(-1, 1, order='F')
            u_shape = shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, xd_pos, n_points * n_points2, shape, R_d, s_d, c_0)
            xd_pos = xd_pos + u_shape * factor * delta
            combined2 = np.hstack((np.reshape(xd_pos, (n_points*n_points2, -1)), np.reshape(factor * u_shape, (n_points*n_points2, -1))))
            xd = combined2.flatten().reshape(-1, 1)

            points = np.array([states[i] for i in quad_indices2])

            if time_num==0:
                flysurf.update(points_coord2, points)
                sampler = FlysurfSampler(flysurf, n_points, points)

            sampler.flysurf.update(points_coord2, points)
            xe_pos = sampler.sampling_v1(fig, ax, points, plot=False)
            combined = np.hstack((xe_pos, vels[indices,:3]))
            xe = combined.flatten().reshape(-1, 1)

            u2 = np.dot(K, (xd-xe))

            u = u2 + u_gravity # Compute control inputs for all drones

            # Enforce actuator limits
            for kv in range(1, n_actuators + 1):
                u[3 * kv - 3] = np.clip(u[3 * kv - 3], u_limits[0, 0], u_limits[0, 1])
                u[3 * kv - 2] = np.clip(u[3 * kv - 2], u_limits[1, 0], u_limits[1, 1])
                u[3 * kv - 1] = np.clip(u[3 * kv - 1], u_limits[2, 0], u_limits[2, 1])

            data.ctrl[:] = u.flatten()
            u_save[:, int(time_num/delta_factor)] = u2.flatten()
            x_save[:, int(time_num/delta_factor)] = x.flatten()
            xe_save[:, int(time_num/delta_factor)] = xe.flatten()
            xd_save[:, int(time_num/delta_factor)] = xd.flatten()

        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        # Render the simulation scene
        if time_num % 2 == 0:
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
            R_d = rotation_matrix(np.pi / 4, 0, 0)
            factor = 0.25

end_time = time.time()  # Record end time
elapsed_time = end_time - start_time  # Calculate elapsed time

print(f"Simulation time: {elapsed_time:.3f} seconds")
t = np.arange(0, iter * delta, delta)
cmap = plt.get_cmap("tab10")

fig = plt.figure(2)
ax = fig.add_subplot(111)
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
plt.figure(3)
plt.plot(t, u_save[2, :], linewidth=1.5)
plt.plot(t, u_save[5, :], linewidth=1.5)
plt.ylabel("Force (N)")
plt.xlabel("Time (s)")
plt.title("Force Fz")
plt.grid(True)

plot_errors(iter, delta, x_save, xd_save, xe_save)

plt.show()

