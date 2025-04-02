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

# SPRING MATRIX AS PARAMETER

# SPRING MATRIX AS PARAMETER

# MODEL PARAMETERS
def main():

    # FLYSURF SIMULATOR PARAMETERS
    rows = 17  # Number of rows (n-1)/(spacing+1)
    cols = rows  # Number of columns
    x_init = -0.5  # Position of point in x (1,1)
    y_init = -0.5  # Position of point in y (1,1)
    x_length = 1  # Total length in x direction
    y_length = 1  # Total length in y direction
    str_stif = 4.0  # Stifness of structural springs
    shear_stif = 4.0  # Stifness of shear springs
    flex_stif = 4.0  # Stifness of flexion springs
    g = 9.81  # Gravity value

    quad_positions = [[1, 1], [rows, 1], [1, cols], [int((rows - 1) / 2) + 1, int((cols - 1) / 2) + 1], [rows, cols],
                      [1, int((cols - 1) / 2) + 1], [int((rows - 1) / 2) + 1, 1], [rows, int((cols - 1) / 2) + 1],
                      [int((rows - 1) / 2) + 1, cols]]  # UAVs positions in the grid simulator
    # quad_positions = [[x, y] for x, y in itertools.product(range(1, rows+1), repeat=2)]
    # quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols]]
    # quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]

    rows2 = 17  # Number of rows (n-1)/(spacing+1)
    cols2 = rows  # Number of columns
    quad_positions2 = [[1, 1], [rows2, 1], [1, cols2], [int((rows2 - 1) / 2) + 1, int((cols2 - 1) / 2) + 1],
                       [rows2, cols2]]
    quad_positions2 = quad_positions

    mass_total = 0.1
    mass_points = mass_total / (rows * cols)  # Mass of each point0
    mass_quads = 0.07  # Mass of each UAV
    damp_point = 0.01  # Damping coefficient on each point
    damp_quad = 0.6  # Damping coefficient on each UAV
    T_s = 0.005  # Simulator step
    u_limits = np.array([[-2.0, 2.0], [-2.0, 2.0], [-0.5, 10.0]])  # Actuator limits
    max_l_str = 0.03  # Maximum elongation from the natural length of the structural springs
    max_l_shear = 2 * max_l_str  # Maximum elongation from the natural length of the shear springs
    max_l_flex = 1.41 * max_l_str  # Maximum elongation from the natural length of the flexion springs
    file_path = "FlySurf_Simulator.xml"  # Output xml file name

    iota_min = 0.5
    iota_max = 1.2

    # Generate xml simulation  file
    [model, data] = generate_xml2(rows, cols, x_init, y_init, x_length, y_length, quad_positions, mass_points,
                                  mass_quads, str_stif, shear_stif, flex_stif, damp_point, damp_quad, T_s, u_limits,
                                  max_l_str, max_l_shear, max_l_flex, file_path)

    spacing_factor = 1
    [x_actuators, n_actuators] = init_simulator(quad_positions, spacing_factor)
    # print('x_actuators', x_actuators)
    # print('n_points:',(rows+spacing_factor)/(spacing_factor+1))

    x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
    y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

    delta_factor = 20
    delta = delta_factor * T_s
    time_change = 5
    n_tasks = 4
    total_time = 52
    time_step_num = round(total_time / T_s)

    n_points = int((rows + spacing_factor) / (spacing_factor + 1))
    n_points2 = int((cols + spacing_factor) / (spacing_factor + 1))
    l0 = (spacing_factor + 1) * x_spacing
    iter = int(time_step_num / delta_factor)

    N_horizon = 5

    [u_save, x_save, xd_save, xe_save, step_time_save, x_gamma_save, u_components_save, xd_sampled, t_save, xd_0_save,
     Rs_d_save, shape_save] = init_vectors2(n_actuators, [rows, cols], iter, [n_points, n_points2], 10 * N_horizon)

    x = np.zeros((n_points * n_points2 * 6, 1))
    for i in range(n_points):
        for j in range(n_points2):
            x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
            x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)
    x[0::6] = x[0::6] - 0.5
    x[1::6] = x[1::6] - 0.5
    x[2::6] = 0

    ld = x_spacing
    xd = np.zeros((rows * cols * 6, 1))
    for i in range(rows):
        for j in range(cols):
            xd[6 * rows * (j) + 6 * (i) + 0] = ld * (i)
            xd[6 * rows * (j) + 6 * (i) + 1] = ld * (j)
    xd[0::6] = xd[0::6] - 0.5
    xd[1::6] = xd[1::6] - 0.5
    xd[2::6] = 0
    xd_iter = xd.copy()

    # CONTROL PARAMETERS
    # Q_vector = [12500000, 12500000, 9500000, 0, 0, 4000, 80000, 4000, 4000, 5000] # [x, y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad, v_y_quad, v_z_quad]
    # R_vector = [32, 32, 32] # [force in x and y, force in z] 40 y 6

    # Q_vector = np.array([10000, 2500000, 0, 0, 500, 600, 15, 10]) # xe [x and y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad and v_y_quad, v_z_quad]
    # R_vector = [7, 8] # [force in x and y, force in z] 40 y 6 xe

    u_gravity = u_gravity_forces(n_UAVs=n_actuators, mass_points=mass_points, mass_UAVs=mass_quads, rows=rows,
                                 cols=cols, g=g)

    # PATH PLANNING PARAMETERS
    alpha_H = 10.0
    alpha_G = 10.0
    alpha_0 = 0.0
    alpha_Hd = 11.0
    shape = np.reshape(np.array([xd[::6], xd[1::6], xd[2::6]]), (3, rows * cols)).reshape(-1, 1, order='F')
    R_d = rotation_matrix(0, 0, 0)
    s_d = 1.0
    c_0 = np.array([0.3, 0.0, 0.55])
    factor = 0.075
    shape_gaussian = shape_gaussian_mesh(sides=[0.9, 0.9], amplitude=1.0, center=[0.0, 0.0], sd=[0.65, 0.65],
                                         n_points=[rows, cols])
    inverted_shape_gaussian = inverted_shape_gaussian_mesh(sides=[0.9, 0.9], amplitude=1.12, center=[0.0, 0.0],
                                                           sd=[0.775, 0.775], n_points=[rows, cols])

    shape_semi_cylinder = shape_semi_cylinder_arc(sides=0.9, amplitude=0.25, center=[0.0, 0.0], radius=0.3,
                                                  n_points=[rows, cols])

    indices = []
    for i in range(1, n_points + 1):
        for j in range(1, n_points2 + 1):
            row_equal = i * (spacing_factor + 1) - spacing_factor
            col_equal = j * (spacing_factor + 1) - spacing_factor
            indices.append((row_equal - 1) * cols + col_equal)
    # print('i',indices)
    indices2 = [i - 1 for i in indices]

    flysurf = CatenaryFlySurf(rows2, cols2, 1 / (rows2 - 1) + 0.0025, num_sample_per_curve=rows2)

    [points_coord2, quad_indices2] = points_coord_estimator(quad_positions, rows, cols)

    [points_coord3, quad_indices3] = points_coord_estimator(quad_positions2, rows2, cols2)
    # print(quad_indices2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_time = time.time()  # Record start time
    time_num = 0

    for ii in range(iter + N_horizon + 1):

        if ii <= iter:
            if 5 >= ii * delta_factor * T_s:
                sep = 5 / (delta_factor * T_s)
                c_0 = np.array([0.0, 0.0, 0.05 + 0.45 * ii / sep])
            if 8.0 == ii * delta_factor * T_s:
                shape = shape_gaussian
            if (16.0 < ii * delta_factor * T_s) and (31 >= ii * delta_factor * T_s):
                sep2 = 16 / (delta_factor * T_s)
                sep3 = 15 / (delta_factor * T_s)
                c_0 = np.array(
                    [0.5 * np.cos(2 * np.pi * (ii - sep2) / sep3) - 0.5, 0.5 * np.sin(2 * np.pi * (ii - sep2) / sep3),
                     0.5])
                yaw = np.arctan2(c_0[0], c_0[1])
                R_d = rotation_matrix(0, 0, -2 * yaw)
            if 34.0 == ii * delta_factor * T_s:
                factor = 0.065
                shape = inverted_shape_gaussian
            if (34.0 < ii * delta_factor * T_s) and (49 >= ii * delta_factor * T_s):
                sep2 = 34 / (delta_factor * T_s)
                sep3 = 15 / (delta_factor * T_s)
                c_0 = np.array(
                    [0.5 * np.cos(2 * np.pi * (ii - sep2) / sep3) - 0.5, 0.5 * np.sin(2 * np.pi * (ii - sep2) / sep3),
                     0.5])
                yaw = np.arctan2(c_0[0], c_0[1])
                R_d = rotation_matrix(0, 0, -yaw)
            if 41.5 == ii * delta_factor * T_s:
                shape = shape_semi_cylinder
            '''
            if (5 * time_change < ii * delta_factor * model.opt.timestep) and (7.0 * time_change >= ii * delta_factor * T_s):
                c_0 = np.array([0.5 * np.cos(1 * np.pi * (ii - sep) / sep)-0.5, 0.5 * np.sin(1 * np.pi * (ii - sep) / sep), 0.65])
            '''
            # if (3.0 * time_change <= ii * delta_factor * T_s) and (5.0 * time_change > ii * delta_factor * T_s):
            #    sep = iter / n_tasks * 2
            #    c_0 = np.array(
            #        [0.3 * np.cos(2 * np.pi * (ii - sep) / sep), 0.3 * np.sin(2 * np.pi * (ii - sep) / sep), 0.45])
            # R_d = rotation_matrix(np.pi/5*np.sin(2*np.pi*(ii-sep)/sep), -np.pi/5*np.cos(2*np.pi*(ii-sep)/sep), 0)
            #    factor = 0.1

            xd_pos = np.reshape(np.array([xd_iter[::6], xd_iter[1::6], xd_iter[2::6]]), (3, rows * cols)).reshape(-1, 1,
                                                                                                                  order='F')
            u_shape = shape_controller_3D_V3(alpha_H, alpha_G, alpha_Hd, xd_pos, rows * cols, shape, np.eye(3), s_d,
                                             c_0)
            xd_pos = xd_pos + u_shape * factor * delta
            xd_pos_vector = np.reshape(xd_pos, (rows * cols, -1))
            u_shape_vector = np.reshape(factor * u_shape, (rows * cols, -1))
            xd_iter = np.hstack((xd_pos_vector, u_shape_vector)).flatten().reshape(-1, 1)

            xd_pos1 = compute_gamma(rows * cols, xd_pos, R_d, s_d, c_0)
            xd_pos_vector2 = np.reshape(xd_pos1, (rows * cols, -1))
            if ii == 0:
                xd_pos_vector2_last = xd_pos_vector2

            u_shape_vector_2 = (xd_pos_vector2 - xd_pos_vector2_last) / (factor * delta)
            combined2 = np.hstack((xd_pos_vector2[indices2], u_shape_vector_2[indices2]))
            xd_iter2 = np.hstack((xd_pos_vector2, 0 * u_shape_vector_2)).flatten().reshape(-1, 1)
            xd = combined2.flatten().reshape(-1, 1)

            xd_pos_vector2_last = xd_pos_vector2

            x_gamma = compute_gamma(rows * cols, shape, R_d, s_d, c_0)

            xd_save[:, ii] = xd_iter2.flatten()
            x_gamma_save[:, ii] = x_gamma.flatten()

        xd_0_save[:, ii] = np.hstack((c_0, 0 * c_0))
        Rs_d_save[:, :, ii] = s_d * R_d
        xd_sampled[:, ii] = xd.flatten()


    positions_d_x = xd_sampled[0::6, :]
    positions_d_y = xd_sampled[1::6, :]
    positions_d_z = xd_sampled[2::6, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(0,positions_d_x.shape[1],4):  # Iterate over iterations
        ax.clear()
        ax.scatter(positions_d_x[:, i], positions_d_y[:, i], positions_d_z[:, i],
                   color='r', marker='x', label='Desired' if i == 0 else "")  # X markers for desired positions

        ax.set_xlim([np.min(positions_d_x), np.max(positions_d_x)])
        ax.set_ylim([np.min(positions_d_y), np.max(positions_d_y)])
        ax.set_zlim([np.min(positions_d_z), np.max(positions_d_z)])

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_zlabel("Z Position")

        ax.set_title(f"Iteration {i}")
        if i == 0:
            ax.legend()

        plt.pause(0.001)  # Adjust pause duration for speed

    t = np.arange(0, (iter + 1) * delta, delta)
    base_directory = "/home/marhes_1/FLYSOM/Data/Simulation"
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_directory = os.path.join(base_directory,
                                        f"STP_MPC_{rows}mesh_{spacing_factor}spacing_{n_actuators}UAV_{experiment_timestamp}")
    os.makedirs(experiment_directory, exist_ok=True)
    plot_errors4(t, iter+1, x_save, xd_save, xe_save,
                 x_gamma_save, experiment_directory)
    plt.show()

if __name__ == "__main__":
    main()
