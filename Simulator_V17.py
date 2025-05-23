import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import sys
import do_mpc
import casadi as ca
import cv2
import glfw

from Generation_Automatic import *
from flysurf_catenary_estimator.catenary_flysurf import *
from util import *
from LQR_MPC_functions import *
import itertools

# SPRING MATRIX AS PARAMETER

# FLYSURF SIMULATOR PARAMETERS
rows = 7 # Number of rows (n-1)/(spacing+1)
cols = rows # Number of columns
x_init = -0.5 # Position of point in x (1,1)
y_init = -0.5 # Position of point in y (1,1)
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.001 # Stifness of structural springs
shear_stif = 0.001 # Stifness of shear springs
flex_stif = 0.001 # Stifness of flexion springs
g = 9.81 # Gravity value
quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols],[1,int((cols-1)/2)+1],[int((rows-1)/2)+1,1],[rows,int((cols-1)/2)+1],[int((rows-1)/2)+1,cols]]  # UAVs positions in the grid simulator
quad_positions = [[x, y] for x, y in itertools.product(range(1, rows+1), repeat=2)]
#quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols]]
#quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]
mass_total = 0.1
mass_points = mass_total/(rows*cols) # Mass of each point0
mass_quads = 0.07 # Mass of each UAV
damp_point = 0.01 # Damping coefficient on each point
damp_quad = 0.6 # Damping coefficient on each UAV
T_s = 0.004 # Simulator step
u_limits = 10*np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.5, 1.0]]) # Actuator limits
file_path = "FlySurf_Simulator.xml"  # Output xml file name

# Generate xml simulation  file
[model, data] = generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, T_s, u_limits*1.01, file_path)

spacing_factor = 0
[x_actuators, n_actuators] = init_simulator(quad_positions, spacing_factor)
#print('x_actuators', x_actuators)
print('n_points:',(rows+spacing_factor)/(spacing_factor+1))

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

delta_factor = 5
delta = delta_factor*T_s
time_change = 2
n_tasks = 2
total_time = time_change*n_tasks +1
time_step_num = round(total_time / T_s)

n_points = int((rows + spacing_factor)/(spacing_factor+1))
n_points2 = int((cols+ spacing_factor)/(spacing_factor+1))
l0= (spacing_factor+1)*x_spacing
iter = int(time_step_num/delta_factor)

N_horizon = 5

[u_save, x_save, xe_save, step_time_save, xd_save, u_components_save, xd_sampled, t_save, xd_0_save, Rs_d_save, shape_save] = init_vectors3(n_actuators, [rows, cols], iter, [n_points, n_points2], N_horizon )

x = np.zeros((n_points * n_points2 * 6,1))
for i in range(n_points):
    for j in range(n_points2):
        x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
        x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)
x[0::6] = x[0::6] - 0.5
x[1::6] = x[1::6] - 0.5
x[2::6] = 0.05

ld= x_spacing
xd = np.zeros((rows*cols* 6,1))
for i in range(rows):
    for j in range(cols):
        xd[6 * rows * (j) + 6 * (i) + 0] = ld * (i)
        xd[6 * rows * (j) + 6 * (i) + 1] = ld * (j)
xd[0::6] = xd[0::6] - 0.5
xd[1::6] = xd[1::6] - 0.5
xd[2::6] = 0.05
xd_iter = xd.copy()

# CONTROL PARAMETERS
Q_vector = [6500, 1500, 0, 0, 6500, 1500, 20, 20] # [x and y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad and v_y_quad, v_z_quad]
R_vector = [2, 2] # [force in x and y, force in z]

u_gravity = u_gravity_forces(n_UAVs = n_actuators, mass_points = mass_points, mass_UAVs = mass_quads, rows =rows, cols=cols, g= g)
#print(u_gravity)30

# PATH PLANNING PARAMETERS
alpha_H = 3
alpha_G = 5
alpha_0 = 10.0
alpha_Hd = 30
shape = np.reshape(np.array([xd[::6],xd[1::6],xd[2::6]]),(3, rows*cols)).reshape(-1,1, order='F')
R_d = rotation_matrix(0, 0, 0)
s_d = 1.0
c_0 = np.array([0.0, 0.0, 0.05])
factor= 0.2
shape_gaussian = shape_gaussian_mesh(sides=[0.85, 0.85], amplitude=1.12, center=[0.0, 0.0], sd = [0.575, 0.575], n_points = [rows, cols])
inverted_shape_gaussian = inverted_shape_gaussian_mesh(sides=[0.8, 0.8], amplitude=1.12, center=[0.0, 0.0], sd = [0.575, 0.575], n_points = [rows, cols])

#shape = shape_gaussian

indices = []
for i in range(1,n_points+1):
    for j in range(1,n_points2+1):
        row_equal = i*(spacing_factor+1) - spacing_factor
        col_equal = j*(spacing_factor+1) - spacing_factor
        indices.append((row_equal-1)*cols+col_equal)
#print('i',indices)
indices2 = [i-1 for i in indices]

flysurf = CatenaryFlySurf(rows, cols, x_spacing + 0.001, num_sample_per_curve=rows)

[points_coord2, quad_indices2] = points_coord_estimator(quad_positions, rows, cols)
#print(quad_indices2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

start_time = time.time()  # Record start time
time_num = 0

for ii in range(iter+N_horizon+1):

    if ii<=iter:

        if time_change >= ii * delta_factor * T_s:
            sep = iter / n_tasks
            c_0 = np.array([0.0 + 0.3*ii/sep, 0.0, 0.05 + 0.5*ii/sep])
        if (time_change < 1.0 * ii* delta_factor *model.opt.timestep) and (2.0 * time_change >= ii * delta_factor * T_s):
            sep = time_change/T_s/delta_factor
            R_d = rotation_matrix(0, 0 + np.pi / 6* (ii - sep) / sep, 0)
            print((ii - sep) / sep)
            shape = shape_gaussian
        #if 2.0 * time_change == ii * delta_factor *model.opt.timestep:
        #    shape = inverted_shape_gaussian
        if (3.0 * time_change <= ii * delta_factor * T_s) and (5.0 * time_change > ii * delta_factor * T_s):
            sep = iter / n_tasks * 2
            c_0 = np.array(
                [0.3 * np.cos(2 * np.pi * (ii - sep) / sep), 0.3 * np.sin(2 * np.pi * (ii - sep) / sep), 0.45])
            # R_d = rotation_matrix(np.pi/5*np.sin(2*np.pi*(ii-sep)/sep), -np.pi/5*np.cos(2*np.pi*(ii-sep)/sep), 0)
            factor = 0.1

        x_gamma = np.reshape(compute_gamma(rows * cols, shape, R_d, s_d, c_0), (rows * cols, -1))
        combined_gamma = np.hstack((x_gamma, 0*x_gamma))
        xd_save[:, ii] = combined_gamma.flatten()

        combined2 = np.hstack((x_gamma[indices2], 0*x_gamma[indices2]))
        xd = combined2.flatten().reshape(-1, 1)

    xd_0_save[:, ii] = np.hstack((c_0,0*c_0))
    Rs_d_save[:, :, ii] =s_d*R_d
    xd_sampled[:, ii] = xd.flatten()

    shape_3 = shape.reshape((3, rows*cols), order='F')
    shape_00 = np.mean(shape_3, axis=1, keepdims=True)  # Centroid of c
    shape_save[:, :, ii] = shape_3-shape_00

mpc_0 = init_MPC_0(str_stif,shear_stif,flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_total/(n_points*n_points2), mass_quads,Q_vector, R_vector, delta, u_limits, g, xd_0_save, N_horizon)
mpc_0.setup()
mpc_0.x0 = x
mpc_0.set_initial_guess()
u_mpc_0 = mpc_0.make_step(x)

mpc_Rs = init_MPC_Rs(str_stif,shear_stif,flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_total/(n_points*n_points2), mass_quads,Q_vector, R_vector, delta, u_limits, g, Rs_d_save, shape_save ,5)
mpc_Rs.setup()
mpc_Rs.x0 = x
mpc_Rs.set_initial_guess()
u_mpc_Rs = mpc_Rs.make_step(x)

mpc_shape = init_MPC_shape(str_stif,shear_stif,flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_total/(n_points*n_points2), mass_quads,Q_vector, R_vector, delta, u_limits, g, Rs_d_save, shape_save ,5)
mpc_shape.setup()
mpc_shape.x0 = x
mpc_shape.set_initial_guess()
u_mpc_shape = mpc_shape.make_step(x)

u_mpc_0_save = u_save.copy()
u_mpc_Rs_save = u_save.copy()
u_mpc_shape_save = u_save.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    if not glfw.init():
        raise Exception("Could not initialize GLFW")

    # Create a window (this creates an OpenGL context)
    view_height, view_width = 720, 720
    window = glfw.create_window(view_height, view_width, "Offscreen", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window")
        
    glfw.make_context_current(window)
    viewer.cam.lookat = [0, -0.65, 1]  # Move camera target in the [x, y, z] direction
    viewer.cam.distance = 2.0  # Zoom out
    viewer.cam.azimuth = 90  # Change azimuth angle
    viewer.cam.elevation = -30  # Change elevation angle
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
    viewport = mujoco.MjrRect(0, 0, view_width, view_height)

    # Create a new scene. The maxgeom parameter specifies the maximum number of geometries.
    scene = mujoco.MjvScene(model, maxgeom=10000)
    # Create a new camera.
    # camera = mujoco.MjvCamera()
    
    # Create a rendering context.
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Create a default options object for the scene.
    mjv_opt = mujoco.MjvOption()
    
    # Allocate a NumPy array to hold the RGB image.
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (view_width, view_height)) 

    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()

        if time_num%delta_factor == 0:
            states = data.xpos
            vels = data.cvel
            combined = np.hstack((states[indices], vels[indices,:3]))
            x = combined.flatten().reshape(-1, 1)

            x_iter = np.hstack((states[1::], vels[1::,:3])).flatten().reshape(-1, 1)

            points = np.array([states[i] for i in quad_indices2])

            if time_num==0:
                flysurf.update(points_coord2, points)
                sampler = FlysurfSampler(flysurf, rows, points)

            sampler.flysurf.update(points_coord2, points)
            all_samples = sampler.sampling_v1(fig, ax, points, plot=False)
            xe_pos = sampler.smooth_particle_cloud(all_samples, 1.0, delta)
            combined = np.hstack((xe_pos[indices2], sampler.vel[indices2]))
            xe = combined.flatten().reshape(-1, 1)

            xe_iter = np.hstack((xe_pos, sampler.vel)).flatten().reshape(-1, 1)

            start_time = time.time()  # Record start time

            #pos = np.reshape(x, (6, n_points*n_points2), order='F')
            #one_1 = np.ones((n_points*n_points2, 1)) /(n_points*n_points2)  # Define an n x 1 column vector
            #pos_0 = (pos @ one_1).flatten()
            #x_b = pos - (pos_0).reshape(6, 1)
            #x_b = x_b[0:3, :]

            #H_var, R_h_var, s_h = matrix_H_3D(x_b, shape_save[:,:,int(time_num/delta_factor)])

            #R_h_var = (R_h_var.T).flatten().tolist()  # Convert to a flat list
            #R_h_var = DM(R_h_var)
            #print(R_h_var)
            #mpc_shape.set_uncertainty_values(R_h=R_h_var)

            #print(np.mean(mpc_shape.data['_p'][int(time_num/delta_factor)]))

            u_mpc_0 = 70*mpc_0.make_step(x)

            u_mpc_Rs = 60*mpc_Rs.make_step(x)

            u_mpc_shape = 10*mpc_shape.make_step(x)

            u_mpc = u_mpc_0 + u_mpc_Rs + u_mpc_shape

            #u_mpc[2::3] = 5*u_mpc[2::3]
            u = u_mpc + u_gravity # Compute control inputs for all drones

            # Enforce actuator limits
            for kv in range(1, n_actuators + 1):
                u[3 * kv - 3] = np.clip(u[3 * kv - 3], u_limits[0, 0], u_limits[0, 1])
                u[3 * kv - 2] = np.clip(u[3 * kv - 2], u_limits[1, 0], u_limits[1, 1])
                u[3 * kv - 1] = np.clip(u[3 * kv - 1], u_limits[2, 0], u_limits[2, 1])

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            data.ctrl[:] = u.flatten()
            step_time_save[0,int(time_num/delta_factor)] = elapsed_time
            u_save[:, int(time_num/delta_factor)] = u_mpc.flatten()
            u_mpc_0_save[:, int(time_num / delta_factor)] = u_mpc_0.flatten()
            u_mpc_Rs_save[:, int(time_num / delta_factor)] = u_mpc_Rs.flatten()
            u_mpc_shape_save[:, int(time_num / delta_factor)] = u_mpc_shape.flatten()
            x_save[:, int(time_num/delta_factor)] = x_iter.flatten()
            xe_save[:, int(time_num/delta_factor)] = xe_iter.flatten()
            t_save[int(time_num/delta_factor)] = time_num*T_s

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

        '''
        if time_num ==1:
            time.sleep(10.0)
        '''

        img = np.empty((view_height, view_width, 3), dtype=np.uint8)
        # === Update the scene ===
        # This fills the scene with the current simulation geometry.
        mujoco.mjv_updateScene(
            model, data,        # model and simulation data
            mjv_opt,            # visualization options
            None,               # no perturbation is applied
            viewer.cam,             # your camera
            mujoco.mjtCatBit.mjCAT_ALL,  # include all categories of geometry
            scene               # the scene to update
        )

        # --- Render the scene ---
        # Render the scene into the offscreen context using the viewer's viewport.
        mujoco.mjr_render(viewport, scene, context)
        # Read the pixels from the current view.
        # The viewer object provides the viewport and context needed.
        mujoco.mjr_readPixels(img, None, viewport, context)
        
        # Since OpenGL’s coordinate system is bottom-up, flip the image vertically.
        img = np.flipud(img)
        output.write(img)

    cv2.destroyAllWindows() 
    output.release() 


t = np.arange(0, (iter+1) * delta, delta)

step = int(time_num/delta_factor)

# Plot 1: Control signal u_z
plot_positions(t_save[0:step-1], x_save[:,0:step-1], xd_save[:,0:step-1], quad_positions, rows, n_actuators)
plot_forces2(t_save[0:step-1], u_mpc_0_save[:,0:step-1], u_mpc_Rs_save[:,0:step-1], u_mpc_shape_save[:,0:step-1], u_save[:,0:step-1])
plot_errors2(t_save[0:step-1], step-1, x_save[:,0:step-1], xd_save[:,0:step-1], xe_save[:,0:step-1], n_tasks)
plot_components(t_save[0:step-1], x_save[:,0:step-1], xd_0_save[:,0:step-1], Rs_d_save[:,:,0:step-1], shape_save[:,:,0:step-1])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(step-1),step_time_save[0,0:step-1].flatten(), marker='o', linestyle='-')
ax.axhline(np.mean(step_time_save[0,0:step-1]), color='r', linestyle='--', linewidth=2, label=f"Mean: {np.mean(step_time_save[0,0:step-1]):.2f}")  # Dashed mean line
ax.set_xlabel("Iteration")
ax.set_ylabel("Time (s)")
ax.set_title("Step Time Over Iterations")
ax.grid(True)
ax.set_ylim(0, 1.1*np.max(step_time_save[0,0:step-1]))
ax.text(step * 0.7, np.max(step_time_save[0,0:step-1]) * 1.02, f"Mean: {np.mean(step_time_save[0,0:step-1]):.3f}", color='k', fontsize=12)


plt.show()

