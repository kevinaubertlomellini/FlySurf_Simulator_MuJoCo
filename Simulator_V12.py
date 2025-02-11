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

# FLYSURF SIMULATOR PARAMETERS
rows = 25 # Number of rows (n-1)/(spacing+1)
cols = rows # Number of columns
x_init = -0.5 # Position of point in x (1,1)
y_init = -0.5 # Position of point in y (1,1)
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.0001 # Stifness of structural springs
shear_stif = 0.0001 # Stifness of shear springs
flex_stif = 0.0001 # Stifness of flexion springs
g = 9.81 # Gravity value
quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols],[1,int((cols-1)/2)+1],[int((rows-1)/2)+1,1],[rows,int((cols-1)/2)+1],[int((rows-1)/2)+1,cols]]  # UAVs positions in the grid simulator
#quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols]]
#quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]
mass_total = 0.12
mass_points = mass_total/(rows*cols) # Mass of each point0
mass_quads = 0.07 # Mass of each UAV
damp_point = 0.01 # Damping coefficient on each point
damp_quad = 0.6 # Damping coefficient on each UAV
T_s = 0.004 # Simulator step
u_limits = 10*np.array([[-1.0, 1.0], [-1.0, 1.0], [-0.25, 1.0]]) # Actuator limits
file_path = "FlySurf_Simulator.xml"  # Output xml file name

# Generate xml simulation  file
[model, data] = generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, T_s, u_limits*1.01, file_path)

spacing_factor = 2
[x_actuators, n_actuators] = init_simulator(quad_positions, spacing_factor)
print('x_actuators', x_actuators)
print('n_points:',(rows+spacing_factor)/(spacing_factor+1))

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

delta_factor = 25
delta = delta_factor*T_s
time_change = 5
n_tasks = 3
total_time = time_change*n_tasks
time_step_num = round(total_time / T_s)

n_points = int((rows + spacing_factor)/(spacing_factor+1))
n_points2 = int((cols+ spacing_factor)/(spacing_factor+1))
l0= (spacing_factor+1)*x_spacing
iter = int(time_step_num/delta_factor)

[u_save, x_save, xd_save, xe_save] = init_vectors(n_actuators, [rows, cols], iter)

x = np.zeros((n_points * n_points2 * 6,1))
for i in range(n_points):
    for j in range(n_points2):
        x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
        x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)
x[0::6] = x[0::6] - 0.5
x[1::6] = x[1::6] - 0.5
x[2::6] = 0

ld= x_spacing
xd = np.zeros((rows*cols* 6,1))
for i in range(rows):
    for j in range(cols):
        xd[6 * rows * (j) + 6 * (i) + 0] = ld * (i)
        xd[6 * rows * (j) + 6 * (i) + 1] = ld * (j)
xd[0::6] = xd[0::6] - 0.5
xd[1::6] = xd[1::6] - 0.5
xd[2::6] = 0
xd_iter = xd.copy()

# CONTROL PARAMETERS
Q_vector = [10000, 10000, 0.2, 0.2] # [x and y, z, velocity in x and y, velocity in z]
R_vector = [1, 2] # [force in x and y, force in z]

u_gravity = u_gravity_forces(n_UAVs = n_actuators, mass_points = mass_points, mass_UAVs = mass_quads, rows =rows, cols=cols, g= g)
#print(u_gravity)

# PATH PLANNING PARAMETERS
alpha_H = 3
alpha_G = 20
alpha_0 = 2.0
alpha_Hd = 3
shape = np.reshape(np.array([xd[::6],xd[1::6],xd[2::6]]),(3, rows*cols)).reshape(-1,1, order='F')
R_d = rotation_matrix(0, 0, 0)
s_d = 1
c_0 = np.array([0.0, 0.0, 0.5])
factor=0.675
shape_gaussian = shape_gaussian_mesh(sides=[0.8, 0.8], amplitude=1.12, center=[0.0, 0.0], sd = [0.575, 0.575], n_points = [rows, cols])

indices = []
for i in range(1,n_points+1):
    for j in range(1,n_points2+1):
        row_equal = i*(spacing_factor+1) - spacing_factor
        col_equal = j*(spacing_factor+1) - spacing_factor
        indices.append((row_equal-1)*cols+col_equal)
print('i',indices)
indices2 = [i-1 for i in indices]

flysurf = CatenaryFlySurf(rows, cols, x_spacing + 0.001, num_sample_per_curve=rows)

[points_coord2, quad_indices2] = points_coord_estimator(quad_positions, rows, cols)
print(quad_indices2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

start_time = time.time()  # Record start time
time_num = 0

mpc = init_MPC_model2(x, str_stif,shear_stif,flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_points*rows*cols/(n_points*n_points2), mass_quads,Q_vector, R_vector, delta, u_limits, g)

p_mpc_template = mpc.get_p_template(n_combinations=1)
p_mpc_template['_p'] = x
#xd[0::6] = xd[0::6] + 0.3
#xd[1::6] = xd[1::6] + 0.2
#xd[2::6] = xd[2::6] + 0.5
mpc.set_p_fun(lambda t_now: p_mpc_template)
mpc.setup()
mpc.x0 = x
mpc.set_initial_guess()
u_mpc = mpc.make_step(x)
#print(u_mpc)

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
    scene = mujoco.MjvScene(model, maxgeom=2000)
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

            xd_pos = np.reshape(np.array([xd_iter[::6], xd_iter[1::6], xd_iter[2::6]]), (3, rows*cols)).reshape(-1, 1, order='F')
            u_shape = shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, xd_pos, rows*cols, shape, R_d, s_d, c_0)
            xd_pos = xd_pos + u_shape * factor * delta
            xd_pos_vector= np.reshape(xd_pos,(rows*cols, -1))
            u_shape_vector = np.reshape(factor * u_shape, (rows*cols, -1))
            xd_iter = np.hstack((xd_pos_vector, u_shape_vector)).flatten().reshape(-1, 1)
            combined2 = np.hstack((xd_pos_vector[indices2], u_shape_vector[indices2]))
            xd = combined2.flatten().reshape(-1, 1)


            points = np.array([states[i] for i in quad_indices2])


            if time_num==0:
                flysurf.update(points_coord2, points)
                sampler = FlysurfSampler(flysurf, rows, points)

            sampler.flysurf.update(points_coord2, points)
            xe_pos = sampler.sampling_v1(fig, ax, points, plot=False)
            combined = np.hstack((xe_pos[indices2], vels[indices,:3]))
            xe = combined.flatten().reshape(-1, 1)

            xe_iter = np.hstack((xe_pos, vels[1::,:3])).flatten().reshape(-1, 1)

            p_mpc_template['_p'] = xd
            mpc.set_p_fun(lambda t_now: p_mpc_template)
            u_mpc = mpc.make_step(xe)

            #u_mpc[2::3] = 5*u_mpc[2::3]
            u = u_mpc + u_gravity # Compute control inputs for all drones

            # Enforce actuator limits
            for kv in range(1, n_actuators + 1):
                u[3 * kv - 3] = np.clip(u[3 * kv - 3], u_limits[0, 0], u_limits[0, 1])
                u[3 * kv - 2] = np.clip(u[3 * kv - 2], u_limits[1, 0], u_limits[1, 1])
                u[3 * kv - 1] = np.clip(u[3 * kv - 1], u_limits[2, 0], u_limits[2, 1])

            data.ctrl[:] = u.flatten()
            u_save[:, int(time_num/delta_factor)] = u_mpc.flatten()
            x_save[:, int(time_num/delta_factor)] = x_iter.flatten()
            xe_save[:, int(time_num/delta_factor)] = xe_iter.flatten()
            xd_save[:, int(time_num/delta_factor)] = xd_iter.flatten()

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
            c_0 = np.array([0.0, 0.0, 0.55])
        if 2.0*time_change == time_num*model.opt.timestep:
            R_d = rotation_matrix(np.pi / 4, 0, 0)
            factor = 0.5
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


t = np.arange(0, iter * delta, delta)

# Plot 1: Control signal u_z
plot_positions(t, x_save, xd_save)
plot_forces(t, u_save)
plot_errors(iter, delta, x_save, xd_save, xe_save)

plt.show()

