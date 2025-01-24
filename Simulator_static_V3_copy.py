import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed
import time
from scipy.optimize import root_scalar
import matplotlib
from scipy.spatial import distance
# matplotlib.use('tkagg')
plt.rc('font', family='serif')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from scipy.spatial import Delaunay
import matplotlib.colors as mcolors

from Generation_Automatic import *
from catenary_flysurf import *

# Define a sequence of colors from green -> gold -> red
christmas_colors = [
    "#006400",  # DarkGreen
    "#FFFFFF",  # White
    "#8B0000"   # DarkRed
]
# Create a linear segmented colormap from these colors
christmas_cmap = mcolors.LinearSegmentedColormap.from_list("christmas", christmas_colors, N=256)


def average_hausdorff_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """
    Computes the average Hausdorff distance between two sets of 3D points.

    Parameters:
        points_a (np.ndarray): An ndarray of shape (N, 3) representing the first set of 3D points.
        points_b (np.ndarray): An ndarray of shape (M, 3) representing the second set of 3D points.

    Returns:
        float: The average Hausdorff distance between the two sets of points.
    """
    if points_a.shape[1] != 3 or points_b.shape[1] != 3:
        raise ValueError("Both input arrays must have 3 columns representing 3D points.")

    # Compute pairwise distances
    d_matrix = distance.cdist(points_a, points_b)

    # Compute directed distances
    d_ab = np.mean(np.min(d_matrix, axis=1))  # Average of minimum distances from A to B
    d_ba = np.mean(np.min(d_matrix, axis=0))  # Average of minimum distances from B to A

    # Average Hausdorff distance
    return (d_ab + d_ba) / 2


def oscillation(i):
    return np.sin(i*np.pi/100)*0.01

def FlySurf_positions(flysurf):
    points_array= np.empty((0, 3))
    for i, surface in enumerate(flysurf.active_surface):
        num_edges = len(surface)
        c, x, y, z = flysurf.catenary_surface_params[surface]
        points = []
        for j in range(num_edges):
            for k in range(j + 1, num_edges):
                edge = tuple(sorted((surface[j], surface[k])))
                for v in flysurf.active_ridge[edge]:
                    good = True
                    for p in points:
                        if np.allclose(v, p):
                            good = False
                            break
                    if good:
                        points.append(v)

        x_mesh, y_mesh = flysurf._generate_mesh_in_triangle(np.array(points[0])[:2],
                                                            np.array(points[1])[:2],
                                                            np.array(points[2])[:2], 20)

        # Compute the fitted z-values
        z_mesh = flysurf._catenary_surface((c, x, y, z), x_mesh, y_mesh)

        points = [x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten()]
        points= np.vstack(points).T
        points_array = np.vstack([points_array, points])
    return points_array

def Euler_distance_points(rows,cols,states):
    states_reshaped = states.reshape((rows, cols, 3))  # Shape (9, 9, 3)

    # Initialize list to store distances
    distances = []

    # Compute distances for adjacent points (right and down)
    for i in range(rows):
        for j in range(cols):
            # Current point
            current_point = states_reshaped[i, j]

            # Right neighbor (if not on the last column)
            if j < cols - 1:
                right_point = states_reshaped[i, j + 1]
                distance = np.linalg.norm(current_point - right_point)
                distances.append(distance)

            # Downward neighbor (if not on the last row)
            if i < rows - 1:
                down_point = states_reshaped[i + 1, j]
                distance = np.linalg.norm(current_point - down_point)
                distances.append(distance)

    # Calculate the average distance
    average_distance = np.mean(distances)

    return average_distance

rows = 25 # Number of rows
cols = 25 # Number of columns
x_init = 0.0
y_init = -0.5
x_length = 1  # Total length in x direction
y_length = 1  # Total length in y direction
str_stif = 0.5
shear_stif = 0.5
flex_stif = 0.5
quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1]]  # List of positions with special elements
#quad_positions = [[rows, cols],[rows, 1],[1, 1],[1, cols]]  # List of positions with special elements
file_path = "config_FlySurf_Simulator.xml"  # Output file name

x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions
points_coord = np.array(quad_positions) - 1
print(points_coord)
quad_indices = []  # Initialize an empty list (vector)
for quad in quad_positions:
    row, col = quad
    # Convert 2D (row, col) to 1D index
    index = (row - 1)*cols + col
    quad_indices.append(index)

generate_xml(rows, cols, x_init, y_init, x_length, y_length, quad_positions, str_stif, shear_stif, flex_stif, file_path)

model = mujoco.MjModel.from_xml_path('scene_FlySurf_Simulator.xml')

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

total_time = 0.1
time_step_num = round(total_time / model.opt.timestep)

time_num = 0
flysurf = CatenaryFlySurf(cols, rows, x_spacing+0.001, num_sample_per_curve=cols+1)

estimated_error = np.zeros([time_step_num,1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and data.time <= total_time:
        step_start = time.time()

        states = data.xpos

        #print(states[1])

        '''
        model.body("quad_1").pos = np.array([0.1, -0.4, 0.45])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.45])
        model.body("quad_3").pos = np.array([0.1, 0.4, 0.45])
        model.body("quad_4").pos = np.array([0.9, 0.4, 0.45])

        '''


        model.body("quad_1").pos = np.array([0.1, -0.4, 0.45])
        model.body("quad_2").pos = np.array([0.9, -0.4, 0.45])
        model.body("quad_3").pos = np.array([0.5, 0.0, 0.45])
        model.body("quad_4").pos = np.array([0.1, 0.4, 0.45])
        model.body("quad_5").pos = np.array([0.9, 0.4, 0.45])


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


mesh_size = 25



flysurf = CatenaryFlySurf(mesh_size, mesh_size, 1.0 / (mesh_size - 1), num_sample_per_curve=mesh_size + 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=90, azim=-90)

for i in range(1):
    points[3, 2] += 0.1 * oscillation(i + 5)

    flysurf.update(points_coord, points)

    all_samples = sampling_v1(fig, ax, flysurf, mesh_size - 1, points)

    # print(all_samples)
    ax.plot(all_samples[0:2, 0], all_samples[0:2, 1], all_samples[0:2, 2], "*")
    ax.plot(all_samples[1:, 0], all_samples[1:, 1], all_samples[1:, 2], "*")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.pause(0.000)
    # input()

'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=60)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

for i in range(100):
    ax.clear()
    ax.view_init(elev=45 + 15 * np.cos(i / 20), azim=60 + 0.5 * i)
    flysurf.update(points_coord, points)
    estimated_points = FlySurf_positions(flysurf)
    print("Average Hausdorff distance error:", average_hausdorff_distance(estimated_points, states))
    visualize(fig, ax, flysurf, plot_dot=False, plot_curve=True, plot_surface=True)
'''