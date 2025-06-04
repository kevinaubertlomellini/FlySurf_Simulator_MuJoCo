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

# SHAPE TRAJECTORY PLANNER + NONLINEAR MPC

# FLYSURF SIMULATOR PARAMETERS
rows = 17 # Number of rows (n-1)/(spacing+1)
cols = rows # Number of columns
x_init = -0.5 # Position of point in x (1,1)
y_init = -0.5 # Position of point in y (1,1)
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.1 # Stifness of structural springs
shear_stif = 0.1 # Stifness of shear springs
flex_stif = 0.1 # Stifness of flexion springs
g = 9.81 # Gravity value

#quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols],[1,int((cols-1)/2)+1],[int((rows-1)/2)+1,1],[rows,int((cols-1)/2)+1],[int((rows-1)/2)+1,cols]]  # UAVs positions in the grid simulator
#quad_positions = [[x, y] for x, y in itertools.product(range(1, rows+1), repeat=2)]
quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols]]
#quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]

rows2 = 17 # Number of rows (n-1)/(spacing+1)
cols2 = rows # Number of columns
quad_positions2 = [[1, 1],[rows2, 1],[1, cols2],[int((rows2-1)/2)+1,int((cols2-1)/2)+1],[rows2, cols2]]
quad_positions2 = quad_positions

mass_total = 0.02
mass_points = mass_total/(rows*cols) # Mass of each point0
mass_quads = 0.04 # Mass of each UAV
damp_point = 0.001 # Damping coefficient on each point
damp_quad = 0.05 # Damping coefficient on each UAV
T_s = 0.005 # Simulator step
u_limits = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]]) # Actuator limits
u_limits2 = np.array([[-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2+mass_quads*9.81]]) # Actuator limits
max_l_str = 0.001  # Maximum elongation from the natural length of the structural springs
max_l_shear = 2*max_l_str  # Maximum elongation from the natural length of the shear springs
max_l_flex = 1.41*max_l_str  # Maximum elongation from the natural length of the flexion springs
file_path = "FlySurf_Simulator.xml"  # Output xml file name

iota_min = 0.5
iota_max = 1.2

# Generate xml simulation  file
[model, data] = generate_xml2(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, T_s, u_limits2, max_l_str, max_l_shear, max_l_flex, file_path)

spacing_factor = 1
[x_actuators, n_actuators] = init_simulator(quad_positions, spacing_factor)
#print('x_actuators', x_actuators)
#print('n_points:',(rows+spacing_factor)/(spacing_factor+1))

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

delta_factor = 20
delta = delta_factor*T_s
time_change = 5
n_tasks = 4
total_time = 50
time_step_num = round(total_time / T_s)

n_points = int((rows + spacing_factor)/(spacing_factor+1))
n_points2 = int((cols+ spacing_factor)/(spacing_factor+1))
l0= (spacing_factor+1)*x_spacing
iter = int(time_step_num/delta_factor)

N_horizon = 5

[u_save, x_save, xd_save, xe_save, step_time_save, x_gamma_save, u_components_save, xd_sampled, t_save, xd_0_save, Rs_d_save, shape_save, shape_sampled_save] = init_vectors4(n_actuators, [rows, cols], iter, [n_points, n_points2], 10*N_horizon )

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
#Q_vector = [12500000, 12500000, 9500000, 0, 0, 4000, 80000, 4000, 4000, 5000] # [x, y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad, v_y_quad, v_z_quad]
#R_vector = [32, 32, 32] # [force in x and y, force in z] 40 y 6

#Q_vector = np.array([10000, 2500000, 0, 0, 500, 600, 15, 10]) # xe [x and y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad and v_y_quad, v_z_quad]
#R_vector = [7, 8] # [force in x and y, force in z] 40 y 6 xe

u_gravity = u_gravity_forces(n_UAVs = n_actuators, mass_points = mass_points, mass_UAVs = mass_quads, rows =rows, cols=cols, g= g)

u_gravity[2]=u_gravity[2]+mass_total/8*g
u_gravity[5]=u_gravity[5]+mass_total/8*g
u_gravity[8]=u_gravity[8]+mass_total/2*g
u_gravity[11]=u_gravity[11]+mass_total/8*g
u_gravity[14]=u_gravity[14]+mass_total/8*g

# PATH PLANNING PARAMETERS
alpha_H = 10.0
alpha_G = 10.0
alpha_0 = 0.0
alpha_Hd = 11.0
shape = np.reshape(np.array([xd[::6],xd[1::6],xd[2::6]]),(3, rows*cols)).reshape(-1,1, order='F')
R_d = rotation_matrix(0, 0, 0)
s_d = 1.0
c_0 = np.array([0.3, 0.0, 0.55])
factor= 0.075
shape_gaussian = shape_gaussian_mesh(sides=[0.9, 0.9], amplitude=1.0, center=[0.0, 0.0], sd = [0.585, 0.585], n_points = [rows, cols])
inverted_shape_gaussian = inverted_shape_gaussian_mesh(sides=[0.9, 0.9], amplitude=1.12, center=[0.0, 0.0], sd = [0.775, 0.775], n_points = [rows, cols])
shape_semi_cylinder = shape_semi_cylinder_arc(sides=0.9, amplitude=1.0, center=[0.0, 0.0], radius=0.32, n_points=[rows, cols])



indices = []
for i in range(1,n_points+1):
    for j in range(1,n_points2+1):
        row_equal = i*(spacing_factor+1) - spacing_factor
        col_equal = j*(spacing_factor+1) - spacing_factor
        indices.append((row_equal-1)*cols+col_equal)
#print('i',indices)
indices2 = [i-1 for i in indices]

flysurf = CatenaryFlySurf(rows2, cols2, 1/(rows2-1) - 0.005 , num_sample_per_curve=rows2)

[points_coord2, quad_indices2] = points_coord_estimator(quad_positions, rows, cols)

[points_coord3, quad_indices3] = points_coord_estimator(quad_positions2, rows2, cols2)
#print(quad_indices2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

start_time = time.time()  # Record start time
time_num = 0


for ii in range(iter+N_horizon+1):

    if ii<=iter:
        if 5 >= ii * delta_factor * T_s:
            sep = 5 / (delta_factor * T_s)
            c_0 = np.array([0.0, 0.0, 0.05 + 0.45 * ii / sep])
        if 10.0 == ii * delta_factor * T_s:
            shape = shape_gaussian
        if (20.0 < ii * delta_factor * T_s) and (35 >= ii * delta_factor * T_s):
            sep2 = 20 / (delta_factor * T_s)
            sep3 = 15 / (delta_factor * T_s)
            c_0 = np.array(
                [0.5 * np.cos(2 * np.pi * (ii - sep2) / sep3) - 0.5, 0.5 * np.sin(2 * np.pi * (ii - sep2) / sep3), 0.5])
            yaw = np.arctan2(c_0[0], c_0[1])
            R_d = rotation_matrix(0, 0, -2 * yaw)
        if 35.0 == ii * delta_factor * T_s:
            factor = 0.065
            shape = inverted_shape_gaussian
        if (35.0 <= ii * delta_factor * T_s) and (50 >= ii * delta_factor * T_s):
            sep2 = 35 / (delta_factor * T_s)
            sep3 = 15 / (delta_factor * T_s)
            c_0 = np.array(
                [0.5 * np.cos(2 * np.pi * (ii - sep2) / sep3) - 0.5, 0.5 * np.sin(2 * np.pi * (ii - sep2) / sep3), 0.5])
            yaw = np.arctan2(c_0[0], c_0[1])
            R_d = rotation_matrix(0, 0, -yaw)
        if 42.5 == ii * delta_factor * T_s:
            shape = shape_semi_cylinder
        '''
        if (5 * time_change < ii * delta_factor * model.opt.timestep) and (7.0 * time_change >= ii * delta_factor * T_s):
            c_0 = np.array([0.5 * np.cos(1 * np.pi * (ii - sep) / sep)-0.5, 0.5 * np.sin(1 * np.pi * (ii - sep) / sep), 0.65])
        '''
        #if (3.0 * time_change <= ii * delta_factor * T_s) and (5.0 * time_change > ii * delta_factor * T_s):
        #    sep = iter / n_tasks * 2
        #    c_0 = np.array(
        #        [0.3 * np.cos(2 * np.pi * (ii - sep) / sep), 0.3 * np.sin(2 * np.pi * (ii - sep) / sep), 0.45])
            # R_d = rotation_matrix(np.pi/5*np.sin(2*np.pi*(ii-sep)/sep), -np.pi/5*np.cos(2*np.pi*(ii-sep)/sep), 0)
        #    factor = 0.1


        xd_pos = np.reshape(np.array([xd_iter[::6], xd_iter[1::6], xd_iter[2::6]]), (3, rows * cols)).reshape(-1, 1,
                                                                                                              order='F')
        u_shape = shape_controller_3D_V3(alpha_H, alpha_G, alpha_Hd, xd_pos, rows * cols, shape, np.eye(3), s_d, c_0)
        xd_pos = xd_pos + u_shape * factor * delta
        xd_pos_vector = np.reshape(xd_pos, (rows * cols, -1))
        u_shape_vector = np.reshape(factor * u_shape, (rows * cols, -1))
        xd_iter = np.hstack((xd_pos_vector, u_shape_vector)).flatten().reshape(-1, 1)


        xd_pos1 = compute_gamma(rows*cols, xd_pos, R_d, s_d, c_0)
        xd_pos_vector2 = np.reshape(xd_pos1, (rows * cols, -1))
        if ii==0:
            xd_pos_vector2_last =  xd_pos_vector2

        u_shape_vector_2 = (xd_pos_vector2 - xd_pos_vector2_last)/(factor*delta)
        combined2 = np.hstack((xd_pos_vector2[indices2], 0*u_shape_vector_2[indices2]))
        xd_iter2 = np.hstack((xd_pos_vector2, 0*u_shape_vector_2)).flatten().reshape(-1, 1)
        xd = combined2.flatten().reshape(-1, 1)

        xd_pos_vector2_last = xd_pos_vector2

        x_gamma = compute_gamma(rows * cols, shape, R_d, s_d, c_0)

        xd_save[:, ii] = xd_iter2.flatten()
        x_gamma_save[:, ii] = x_gamma.flatten()


    xd_0_save[:, ii] = np.hstack((c_0, 0 * c_0))
    Rs_d_save[:, :, ii] = s_d * R_d
    xd_sampled[:, ii] = xd.flatten()

    shape_3 = xd_pos.reshape((3, rows * cols), order='F')
    shape_00 = np.mean(shape_3, axis=1, keepdims=True)  # Centroid of c
    shape_sampled_save[:, :, ii] = ((shape_3 - shape_00).T)[indices2].T
    shape_save[:, :, ii] = shape_3 - shape_00

spring_factor = 25

alpha_H = 0
alpha_G = 0
alpha_Rs = 0
Q_0 = 11000 * np.eye(6)
Q_0[0:2, 0:2] = 1200 * np.eye(2)
Q_0[3:5, 3:5] = 0 * np.eye(2)
Q_0[5, 5] = 0
R_vector = 10*np.array([8, 60])

mpc_0 = init_MPC_general(x,1/spring_factor*str_stif,1/spring_factor*shear_stif,1/spring_factor*flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_points*rows*cols/(n_points*n_points2), mass_quads,Q_0, alpha_H , alpha_G, alpha_Rs,  R_vector, delta, u_limits, g, Rs_d_save, shape_sampled_save, xd_0_save, N_horizon, iota_min, iota_max)
mpc_0.setup()
mpc_0.x0 = x
mpc_0.set_initial_guess()

alpha_H = 30
alpha_G = 70*0
alpha_Rs = 9500
Q_0 = 0 * np.eye(6)
R_vector =50*np.array([750, 2400])

mpc_Rs = init_MPC_general(x,1/spring_factor*str_stif,1/spring_factor*shear_stif,1/spring_factor*flex_stif,damp_point,damp_quad,l0,n_points, n_points2, n_actuators, x_actuators, mass_points*rows*cols/(n_points*n_points2), mass_quads,Q_0, alpha_H , alpha_G, alpha_Rs,  R_vector, delta, u_limits, g, Rs_d_save, shape_sampled_save, xd_0_save, N_horizon, iota_min, iota_max)
mpc_Rs.setup()
mpc_Rs.x0 = x
mpc_Rs.set_initial_guess()

u_mpc = mpc_0.make_step(x) + mpc_Rs.make_step(x)

est_t_save = step_time_save.copy()

with (mujoco.viewer.launch_passive(model, data) as viewer):
    if not glfw.init():
        raise Exception("Could not initialize GLFW")

    # Create a window (this creates an OpenGL context)
    view_height, view_width = 1720, 1720
    window = glfw.create_window(view_height, view_width, "Offscreen", None, None)
    if not window:
        glfw.terminate()
        raise Exception("Could not create GLFW window")

    glfw.make_context_current(window)
    viewer.cam.lookat = [-0.5, -0.75, 1]  # Move camera target in the [x, y, z] direction
    viewer.cam.distance = 2.4  # Zoom out
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
    output = cv2.VideoWriter("SMPC_Mujoco_5UAV.avi", cv2.VideoWriter_fourcc(*'MPEG'), 1 / T_s,
                             (view_width, view_height))

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
                flysurf.update(points_coord3, points)
                sampler = FlysurfSampler(flysurf, rows, points, points_coord3)

            start_time = time.time()

            sampler.flysurf.update(points_coord3, points)
            all_samples = sampler.sampling_v1(fig, ax, points, coordinates=points_coord3, plot=False)
            xe_pos = sampler.smooth_particle_cloud(all_samples, 1.0, delta)
            combined = np.hstack((xe_pos[indices2], sampler.vel[indices2]))
            combined = np.hstack((xe_pos[indices2], vels[indices,:3]))
            xe = combined.flatten().reshape(-1, 1)

            xe_iter = np.hstack((xe_pos, sampler.vel)).flatten().reshape(-1, 1)

            elapsed_time = time.time() - start_time
            est_t_save[0, int(time_num / delta_factor)] = elapsed_time

            start_time = time.time()  # Record start time

            u_mpc = mpc_0.make_step(xe) + mpc_Rs.make_step(xe)

            #u_mpc[2::3] = 5*u_mpc[2::3]

            # Enforce actuator limits
            #for kv in range(1, n_actuators + 1):
            #    u_mpc[3 * kv - 3] = np.clip(u_mpc[3 * kv - 3], u_limits[0, 0], u_limits[0, 1])
            #    u_mpc[3 * kv - 2] = np.clip(u_mpc[3 * kv - 2], u_limits[1, 0], u_limits[1, 1])
            #    u_mpc[3 * kv - 1] = np.clip(u_mpc[3 * kv - 1], u_limits[2, 0], u_limits[2, 1])

            u = u_mpc + u_gravity  # Compute control inputs for all drones

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time

            data.ctrl[:] = u.flatten()
            step_time_save[0,int(time_num/delta_factor)] = elapsed_time
            u_save[:, int(time_num/delta_factor)] = u_mpc.flatten()
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

base_directory = "/home/marhes_1/FLYSOM/Data/Simulation"
experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_directory = os.path.join(base_directory,f"SMPC_{rows}mesh_{spacing_factor}spacing_{n_actuators}UAV_{experiment_timestamp}")
os.makedirs(experiment_directory, exist_ok=True)

plot_positions(t_save[0:step-1], x_save[:,0:step-1], xd_save[:,0:step-1], quad_positions, rows, n_actuators, experiment_directory)
plot_forces(t_save[0:step-1], u_save[:,0:step-1], experiment_directory)
plot_errors4(t_save[0:step-1], step-1, x_save[:,0:step-1], xd_save[:,0:step-1], xe_save[:,0:step-1], x_gamma_save[:,0:step-1], experiment_directory)
plot_components(t_save[0:step-1], xe_save[:,0:step-1], xd_0_save[:,0:step-1], Rs_d_save[:,:,0:step-1], shape_save[:,:,0:step-1], experiment_directory)
plot_step_time(step-1, step_time_save[:,0:step], "Step Time Over Iterations- Controller", "controller_step_time_plot.png", experiment_directory)
plot_step_time(step-1, est_t_save[:,0:step], "Step Time Over Iterations- Estimator", "estimator_step_time_plot.png", experiment_directory)

# SAVE DATA
save_data(rows, cols, spacing_factor, n_actuators, step, xe_save, xd_save, x_gamma_save, x_save, xd_0_save, Rs_d_save, shape_save, u_save, t_save, step_time_save, est_t_save,experiment_directory)

plt.show()
