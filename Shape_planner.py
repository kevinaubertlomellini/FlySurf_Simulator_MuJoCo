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
import os
import pandas as pd
from scipy.linalg import expm
from datetime import datetime, timedelta

from flysurf_catenary_estimator.catenary_flysurf import *
from util2 import *
from LQR_MPC_functions import *

# SPRING MATRIX AS PARAMETER

# SPRING MATRIX AS PARAMETER

# MODEL PARAMETERS
def main():
    rows = 7 # Number of rows (n-1)/(spacing+1)
    cols = rows # Number of columns
    x_init = -0.3 # Position of point in x (1,1)
    y_init = -0.3 # Position of point in y (1,1)
    x_length = 0.7# Total length in x direction
    y_length = 0.7  # Total length in y direction
    str_stif = 0.01 # Stifness of structural springs
    shear_stif = 0.01 # Stifness of shear springs
    flex_stif = 0.01 # Stifness of flexion springs
    g = 9.81 # Gravity value
    #quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols],[1,int((cols-1)/2)+1],[int((rows-1)/2)+1,1],[rows,int((cols-1)/2)+1],[int((rows-1)/2)+1,cols]]  # UAVs positions in the grid simulator
    # quad_positions = [[1, 1],[rows, 1],[1, cols],[int((rows-1)/2)+1,int((cols-1)/2)+1],[rows, cols]]
    quad_positions = [[1, 1],[rows, 1],[1, cols],[rows, cols]]
    mass_total = 0.05
    mass_points = mass_total/(rows*cols) # Mass of each point0
    mass_quads = 0.035 # Mass of each UAV
    damp_point = 0.01 # Damping coefficient on each point
    damp_quad = 0.01 # Damping coefficient on each UAV
    T_s = 0.004 # Simulator step
    u_limits = np.array([[-0.01, 0.01], [-0.01, 0.01], [-0.4, 0.4]]) # Actuator limits in N: [F_x, F_y, F_z]

    spacing_factor = 0
    [x_actuators, n_actuators] = init_simulator(quad_positions, spacing_factor)
    print('n_points:',(rows+spacing_factor)/(spacing_factor+1))

    x_spacing = x_length / (cols - 1)  # Adjusted for the correct number of divisions
    y_spacing = y_length / (rows - 1)  # Adjusted for the correct number of divisions

    delta_factor = 25
    delta = delta_factor*T_s
    time_change = 25
    n_tasks = 5
    total_time = time_change*n_tasks
    time_step_num = round(total_time / T_s)

    n_points = int((rows + spacing_factor)/(spacing_factor+1))
    n_points2 = int((cols+ spacing_factor)/(spacing_factor+1))
    l0= (spacing_factor+1)*x_spacing
    iter = int(time_step_num/delta_factor)

    N_horizon = 5

    [u_save, x_save, xd_save, xe_save, step_time_save, x_gamma_save, u_components_save, xd_sampled, t_save] = init_vectors2(n_actuators, [rows, cols], iter, [n_points, n_points2], N_horizon )

    x = np.zeros((n_points * n_points2 * 6,1))
    for i in range(n_points):
        for j in range(n_points2):
            x[6 * n_points * (j) + 6 * (i) + 0] = l0 * (i)
            x[6 * n_points * (j) + 6 * (i) + 1] = l0 * (j)
    x[0::6] = x[0::6] - x_length/2
    x[1::6] = x[1::6] - x_length/2
    x[2::6] = 0.3

    ld= x_spacing
    xd = np.zeros((rows*cols* 6,1))
    for i in range(rows):
        for j in range(cols):
            xd[6 * rows * (j) + 6 * (i) + 0] = ld * (i)
            xd[6 * rows * (j) + 6 * (i) + 1] = ld * (j)
    xd[0::6] = xd[0::6] - x_length/2
    xd[1::6] = xd[1::6] - x_length/2
    xd[2::6] = 0.3
    xd_iter = xd.copy()

    # CONTROL PARAMETERS
    Q_vector = [900, 900, 0.0, 0.0, 900, 900, 120, 120] # [x and y, z, v_x and v_y, v_z, x_UAV and y_UAV, z_UAV , v_x_quad and v_y_quad, v_z_quad]
    R_vector = [2, 7] # [force in x and y, force in z]

    u_gravity = u_gravity_forces(n_UAVs = n_actuators, mass_points = mass_points, mass_UAVs = mass_quads, rows =rows, cols=cols, g= g)
    #print(u_gravity)

    # PATH PLANNING PARAMETERS
    alpha_H = 3  # 3
    alpha_G = 5  # 5
    alpha_0 = 5.0  # 10.0
    alpha_Hd = 8.0  # 30
    shape = np.reshape(np.array([xd[::6], xd[1::6], xd[2::6]]), (3, rows * cols)).reshape(-1, 1, order='F')
    R_d = rotation_matrix(0, 0, 0)
    s_d = 1.0
    c_0 = np.array([0.0, 0.0, 0.5])
    factor = 0.045
    shape_gaussian = shape_gaussian_mesh(sides=[0.7 * x_length, 0.7 * x_length], amplitude=1.12, center=[0.0, 0.0],
                                         sd=[1.0, 1.0], n_points=[rows, cols])
    inverted_shape_gaussian = inverted_shape_gaussian_mesh(sides=[0.8 * x_length, 0.8 * x_length], amplitude=1.2,
                                                           center=[0.0, 0.0], sd=[1.2, 1.2], n_points=[rows, cols])

    shape = inverted_shape_gaussian

    indices = []
    for i in range(1,n_points+1):
        for j in range(1,n_points2+1):
            row_equal = i*(spacing_factor+1) - spacing_factor
            col_equal = j*(spacing_factor+1) - spacing_factor
            indices.append((row_equal-1)*cols+col_equal)
    #print('i',indices)
    indices2 = [i-1 for i in indices]

    flysurf = CatenaryFlySurf(rows, cols, x_spacing + 0.002, num_sample_per_curve=rows)

    [points_coord2, quad_indices2] = points_coord_estimator(quad_positions, rows, cols)
    #print(quad_indices2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    start_time = time.time()  # Record start time
    time_num = 0

    for ii in range(iter+N_horizon+1):

        if ii<=iter:
            xd_pos = np.reshape(np.array([xd_iter[::6], xd_iter[1::6], xd_iter[2::6]]), (3, rows * cols)).reshape(-1, 1,
                                                                                                                  order='F')
            [u_shape, u_components_save[:, ii]] = shape_controller_3D(alpha_H, alpha_G, alpha_0,
                                                                                                alpha_Hd, xd_pos, rows * cols,
                                                                                                shape, R_d, s_d, c_0)
            xd_pos = xd_pos + u_shape * factor * delta
            xd_pos_vector = np.reshape(xd_pos, (rows * cols, -1))
            u_shape_vector = np.reshape(factor * u_shape, (rows * cols, -1))
            xd_iter = np.hstack((xd_pos_vector, u_shape_vector)).flatten().reshape(-1, 1)
            combined2 = np.hstack((xd_pos_vector[indices2], u_shape_vector[indices2]))
            xd = combined2.flatten().reshape(-1, 1)

            x_gamma = compute_gamma(rows * cols, shape, R_d, s_d, c_0)


            xd_save[:, ii] = xd_iter.flatten()
            x_gamma_save[:, ii] = x_gamma.flatten()

            if time_change == 1.0 * ii * delta_factor * T_s:
                R_d = rotation_matrix(0, -np.pi / 5, 0)
                c_0 = np.array([0.75, 0.0, 0.75])
                s_d = 1.0

            if (2.0 * time_change <= ii * delta_factor * T_s) and (4.0 * time_change > ii * delta_factor * T_s):
                sep = iter / n_tasks * 2
                c_0 = np.array(
                    [0.75 * np.cos(2 * np.pi * (ii - sep) / sep), 0.75 * np.sin(2 * np.pi * (ii - sep) / sep), 0.75])
                yaw = np.arctan2(c_0[1], c_0[0])
                R_d = rotation_matrix(0, -np.pi / 5, yaw)
                factor = 0.18
            '''
            if time_change == ii * delta_factor *T_s:
                R_d = rotation_matrix(0, 0, 0)
                c_0 = np.array([-0.3, 0.0, 0.4])
                factor = 0.05
            '''

            if 4 * time_change == ii * delta_factor * T_s:
                c_0 = np.array([0.0, 0.0, 0.25])
                R_d = rotation_matrix(0, 0, 0)
                factor = 0.045


        xd_sampled[:, ii] = xd.flatten()


    positions_d_x = xd_sampled[0::6, :]
    positions_d_y = xd_sampled[1::6, :]
    positions_d_z = xd_sampled[2::6, :]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(positions_d_x.shape[1]):  # Iterate over iterations
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

if __name__ == "__main__":
    main()
