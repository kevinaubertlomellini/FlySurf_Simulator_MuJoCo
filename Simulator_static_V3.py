import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='serif')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from Generation_Automatic import *
from catenary_flysurf import *
from util import *

rows = 32 # Number of rows
cols = 32 # Number of columns
x_init = 0.0
y_init = -0.5
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.5
shear_stif = 0.5
flex_stif = 0.5
quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1]]  # List of positions with special elements
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols]]  # List of positions with special elements
mass_points = 0.001
mass_quads = 0.032
damp_point = 0.01
damp_quad = 0.06
file_path = "config_FlySurf_Simulator.xml"  # Output file name

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions
points_coord = np.array(quad_positions) - 1
quad_indices = func_quad_indices(quad_positions, cols)

generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points, mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, file_path)
model = mujoco.MjModel.from_xml_path('scene_FlySurf_Simulator.xml')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

total_time = 5
time_step_num = round(total_time / model.opt.timestep)

time_num = 0
flysurf = CatenaryFlySurf(cols, rows, x_spacing+0.0011, num_sample_per_curve=cols+1)

estimated_error = np.zeros([time_step_num,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()

        states = data.xpos

        #print(states[1])
        '''
        model.body("quad_1").pos = np.array([0.1, -0.4, 0.65])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.65])
        model.body("quad_3").pos = np.array([0.1, 0.4, 0.65])
        model.body("quad_4").pos = np.array([0.9, 0.4, 0.65])



        '''
        model.body("quad_1").pos = np.array([0.1, -0.4, 0.65])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.65])
        model.body("quad_3").pos = np.array([0.5, 0.0, 0.65])
        model.body("quad_4").pos = np.array([0.1, 0.4, 0.65])
        model.body("quad_5").pos = np.array([0.9, 0.4, 0.65])


        '''
        model.body("quad_1").pos = np.array([0.1, -0.4+ data.time / 75, 0.45 + data.time / 100])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.45 + data.time / 100])
        model.body("quad_3").pos = np.array([0.5, 0.0, 0.45])
        model.body("quad_4").pos = np.array([0.1, 0.4, 0.45])
        model.body("quad_5").pos = np.array([0.9, 0.4, 0.45])
        '''


        '''
        model.body("quad_1").pos = np.array([0.1, -0.4, 0.45])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.45])
        model.body("quad_3").pos = np.array([0.5, 0.0, 0.45])
        model.body("quad_5").pos = np.array([0.1, 0.4, 0.45])
        model.body("quad_6").pos = np.array([0.9, 0.4, 0.45])
        '''

        '''
        model.body("quad_1").pos = np.array([0.1 + data.time / 100, -0.4 + data.time / 75, 0.45 + data.time / 100])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.35 + data.time / 100])
        model.body("quad_3").pos = np.array([0.1, 0.4, 0.35 + data.time / 100])
        model.body("quad_4").pos = np.array([0.9, 0.4, 0.35 + data.time / 100])
        '''
        points = np.array([states[i] for i in quad_indices])

        mujoco.mj_step(model, data)

        flysurf.update(points_coord, points)
        estimated_points = sampling_v1(fig,ax, flysurf, cols-1, points, plot=False)

        #visualize(fig, ax, flysurf, plot_dot=True, plot_curve=True, plot_surface=True, num_samples=10)
        '''
        estimated_points2 = estimated_points.copy()
        estimated_points2[:, 2] = estimated_points2[:, 2] + 0.1
        print(estimated_points[3, 2])
        print(estimated_points2[3, 2])
        '''
        estimated_error[time_num] = average_hausdorff_distance(states[1:], estimated_points)
        #print("Average Hausdorff distance error:", estimated_error[time_num])


        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        time_num += 1
        if time_num >= time_step_num:
            break

average_distance = Euler_distance_points(rows,cols,states[1:])
print(abs(average_distance - x_spacing))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(estimated_error)
ax.set_xlabel("X")


fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

# Scatter plot of points
ax2.scatter(states[1:, 0], states[1:, 1], states[1:, 2], c='b')
ax2.scatter(estimated_points[:, 0], estimated_points[:, 1], estimated_points[:, 2], c='r')

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

# Scatter plot of points
ax2.scatter(estimated_points[:, 0], estimated_points[:, 1], estimated_points[:, 2], c='r')

# Labels and title
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.set_title('3D Plot of Points')
plt.show()
