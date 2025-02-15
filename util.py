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

def func_quad_indices(quad_positions, cols):
    quad_indices = []  # Initialize an empty list (vector)
    for quad in quad_positions:
        row, col = quad
        # Convert 2D (row, col) to 1D index
        index = (row - 1) * cols + col
        quad_indices.append(index)
    return quad_indices

def shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, x, n_points, c, R_d, s_d, c_0):
    """
    Python version of the shape_controller_3D function.

    Parameters:
    alpha_H, alpha_G, alpha_0, alpha_Hd : float
        Control parameters.
    x : numpy.ndarray
        Current positions, reshaped as a (3, n_points) array.
    n_points : int
        Number of points.
    c : numpy.ndarray
        Desired positions, reshaped as a (3, n_points) array.
    R_d : numpy.ndarray
        Desired rotation matrix (3x3).
    s_d : float
        Desired scaling factor.
    c_0 : numpy.ndarray
        Desired centroid position (3,).

    Returns:
    numpy.ndarray
        Control input reshaped as a (3*n_points,) array.
    """
    # Reshape x and c to (3, n_points)
    x = x.reshape((3, n_points), order='F')
    c = c.reshape((3, n_points), order='F')
    # Compute centroids
    x_0 = np.mean(x, axis=1, keepdims=True)  # Centroid of x
    c_00 = np.mean(c, axis=1, keepdims=True)  # Centroid of c

    # Compute shapes (subtract centroids)
    x_b = x - x_0
    c_b = c - c_00

    # Compute H, R_h, s_h using matrix_H_3D function
    H, R_h, s_h = matrix_H_3D(x_b, c_b)

    # Control laws
    u_H = H @ c_b - x_b
    G = x_b @ np.linalg.pinv(c_b)
    u_G = G @ c_b - x_b
    u_y = alpha_H * u_H + alpha_G * u_G
    u_0 = -alpha_0 * (x_0 - c_0.reshape((3, 1))) @ np.ones((1, n_points))
    Hd = s_d * R_d
    u_Hd = alpha_Hd * (Hd @ c_b - x_b)

    # Total control input
    u = u_y + u_0 + u_Hd
    # Reshape to (3*n_points,)
    return [u.flatten(order='F').reshape(-1, 1), np.array([np.mean(np.abs(alpha_H*u_H)),np.mean(np.abs(alpha_G*u_G)),np.mean(np.abs(u_0)),np.mean(np.abs(u_Hd))])]

def compute_gamma(n_points, c, R_d, s_d, c_0):
    c = c.reshape((3, n_points), order='F')
    c_00 = np.mean(c, axis=1, keepdims=True)  # Centroid of c
    c_b = c - c_00
    x_gamma = s_d * R_d @ c_b + c_0.reshape(3, 1)
    return x_gamma.flatten(order='F').reshape(-1, 1)


def matrix_H_3D(Q_b, C_b):
    """
    Compute the transformation matrix H, rotation matrix R_h, and scaling factor s_h
    for 3D point sets Q_b and C_b.

    Parameters:
    Q_b (numpy.ndarray): A 3xN matrix representing the centered coordinates of the first point set.
    C_b (numpy.ndarray): A 3xN matrix representing the centered coordinates of the second point set.

    Returns:
    H (numpy.ndarray): The transformation matrix combining rotation and scaling.
    R_h (numpy.ndarray): The rotation matrix.
    s_h (float): The scaling factor.
    """
    # Compute the covariance matrix
    H_R = C_b @ Q_b.T

    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H_R)

    # Compute the rotation matrix
    R_h = Vt.T @ U.T

    # Compute the scaling factor
    c_s = np.trace(C_b @ C_b.T)
    s_h = np.trace(Q_b.T @ R_h @ C_b) / c_s

    # Compute the transformation matrix
    H = s_h * R_h

    return H, R_h, s_h

def x_des(n_points, n_points2, c, r_d, s_d, c_0):
    # Reshape c into a 3 x (n_points * n_points2) matrix
    c2 = np.reshape(c, (3, n_points * n_points2))

    # Subtract the centroid of c
    c2 = c2 - np.array([
        np.mean(c[0::3]),
        np.mean(c[1::3]),
        np.mean(c[2::3])
    ]).reshape(-1, 1)

    # Compute x_c
    x_c = s_d * r_d @ c2 + c_0[:, np.newaxis]

    # Reshape x_c into a 1D array (column vector equivalent)
    x_c = x_c.flatten()

    return x_c


def rotation_matrix(roll, pitch, yaw):
    # Compute individual rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])

    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R

def init_simulator(quad_positions, spacing_factor):
    quad_positions_ordered = sorted(quad_positions, key=lambda x: (x[1], x[0]))
    print(quad_positions_ordered)
    x_actuators = ((np.array(quad_positions_ordered)+spacing_factor)/(spacing_factor+1)).astype(int)
    n_actuators = x_actuators.shape[0]
    return [x_actuators, n_actuators]

def shape_gaussian_mesh(sides, amplitude, center, sd, n_points):
    # Generate mesh grid for x and y coordinates
    x_g = np.linspace(-sides[0]/2, sides[0]/2, n_points[0])
    y_g = np.linspace(-sides[1]/2, sides[1]/2, n_points[1])
    X_g, Y_g = np.meshgrid(x_g, y_g)

    x_g_vector0 = X_g.flatten()
    y_g_vector0 = Y_g.flatten()

    # Define Gaussian parameters
    Amp = amplitude # Amplitude
    x0 = center[0] # Center of Gaussian in x
    y0 = center[1] # Center of Gaussian in y
    sigma_x = sd[0]  # Standard deviation in x
    sigma_y = sd[1] # Standard deviation in y

    # Calculate the 2D Gaussian
    z_g_vector0 = Amp * np.exp(-((x_g_vector0 - x0) ** 2 / (2 * sigma_x ** 2) +
                                 (y_g_vector0 - y0) ** 2 / (2 * sigma_y ** 2))) - 0.5

    # Combine into shape_gaussian
    shape_gaussian = np.vstack((x_g_vector0, y_g_vector0, z_g_vector0)).T.flatten()
    return shape_gaussian


def inverted_shape_gaussian_mesh(sides, amplitude, center, sd, n_points):
    # Generate mesh grid for x and y coordinates
    x_g = np.linspace(-sides[0]/2, sides[0]/2, n_points[0])
    y_g = np.linspace(-sides[1]/2, sides[1]/2, n_points[1])
    X_g, Y_g = np.meshgrid(x_g, y_g)

    x_g_vector0 = X_g.flatten()
    y_g_vector0 = Y_g.flatten()

    # Define Gaussian parameters
    Amp = amplitude # Amplitude
    x0 = center[0] # Center of Gaussian in x
    y0 = center[1] # Center of Gaussian in y
    sigma_x = sd[0]  # Standard deviation in x
    sigma_y = sd[1] # Standard deviation in y

    # Calculate the 2D Gaussian
    z_g_vector0 = -Amp * np.exp(-((x_g_vector0 - x0) ** 2 / (2 * sigma_x ** 2) +
                                 (y_g_vector0 - y0) ** 2 / (2 * sigma_y ** 2))) - 0.5

    # Combine into shape_gaussian
    shape_gaussian = np.vstack((x_g_vector0, y_g_vector0, z_g_vector0)).T.flatten()
    return shape_gaussian

def u_gravity_forces(n_UAVs, mass_points, mass_UAVs , rows, cols, g):
    u_Forces = np.zeros((3 * n_UAVs, 1))
    for kv in range(1, n_UAVs + 1):
        forces = np.array([0, 0, ((cols * rows) * mass_points / n_UAVs + mass_UAVs - mass_points * n_UAVs) * g])
        #forces = np.array([0, 0, mass_points*g])
        '''
        if kv != 3:
            forces = np.array([0, 0, ((cols * rows) * mass_points / 8 + mass_UAVs- mass_points * n_UAVs) * g])
        else:
            forces = np.array([0, 0, ((cols * rows) * mass_points / 2 + mass_UAVs- mass_points * n_UAVs) * g])
        '''
        u_Forces[3 * kv - 3:3 * kv, 0] = forces.flatten()
    return u_Forces

def init_vectors(n_actuators, n_points, iter ):
    u_save = np.zeros((3 * n_actuators, iter))
    x_save = np.zeros((6 * n_points[0] * n_points[1], iter))
    x_gamma_save = np.zeros((3 * n_points[0] * n_points[1], iter))
    xd_save = np.zeros((6 * n_points[0] * n_points[1], iter))
    xe_save = np.zeros((6 * n_points[0] * n_points[1], iter))
    step_time_save = np.zeros((1, iter))
    u_components_save = np.zeros((4, iter))
    return [u_save, x_save, xd_save, xe_save,step_time_save, x_gamma_save, u_components_save]


def init_vectors2(n_actuators, n_points, iter, n_points_sampled, N_horizon ):
    u_save = np.zeros((3 * n_actuators, iter+1))
    x_save = np.zeros((6 * n_points[0] * n_points[1], iter+1))
    x_gamma_save = np.zeros((3 * n_points[0] * n_points[1], iter+1))
    xd_save = np.zeros((6 * n_points[0] * n_points[1], iter+1))
    xe_save = np.zeros((6 * n_points[0] * n_points[1], iter+1))
    step_time_save = np.zeros((1, iter+1))
    u_components_save = np.zeros((4, iter+1))
    xd_sampled = np.zeros((6 * n_points_sampled[0] * n_points_sampled[1], iter+N_horizon+1))
    return [u_save, x_save, xd_save, xe_save,step_time_save, x_gamma_save, u_components_save, xd_sampled]

def plot_errors(iter, delta, x_save, xd_save, xe_save, x_gamma, n_tasks):
    t = np.arange(0, iter * delta, delta)

    e_save = np.zeros((iter, 1))
    e_save2 = np.zeros((iter, 1))
    e_est_save = np.zeros((iter, 1))
    e_est_save2 = np.zeros((iter, 1))
    e_real_save = np.zeros((iter, 1))
    e_real_save2 = np.zeros((iter, 1))
    e_gamma_beta = np.zeros((iter, 1))
    e_gamma_beta2 = np.zeros((iter, 1))
    e_gamma_des = np.zeros((iter, 1))
    e_gamma_des2 = np.zeros((iter, 1))

    # = iter*delta/n_tasks

    for i in range(iter):
        pos = np.array([x_save[::6, i], x_save[1::6, i], x_save[2::6, i]]).T
        pos_e = np.array([xe_save[::6, i], xe_save[1::6, i], xe_save[2::6, i]]).T
        pos_d = np.array([xd_save[::6, i], xd_save[1::6, i], xd_save[2::6, i]]).T
        pos_gamma = np.array([x_gamma[::3, i], x_gamma[1::3, i], x_gamma[2::3, i]]).T
        e_save[i] = average_hausdorff_distance(pos_e, pos_d)
        e_save2[i] = np.mean(np.linalg.norm(pos_e - pos_d, axis=1))
        e_est_save[i] = average_hausdorff_distance(pos_e, pos)
        e_est_save2[i] = np.mean(np.linalg.norm(pos_e - pos, axis=1))
        e_real_save[i] = average_hausdorff_distance(pos, pos_d)
        e_real_save2[i] = np.mean(np.linalg.norm(pos - pos_d, axis=1))
        e_gamma_beta[i] = average_hausdorff_distance(pos, pos_gamma)
        e_gamma_beta2[i] = np.mean(np.linalg.norm(pos - pos_gamma, axis=1))
        e_gamma_des[i] = average_hausdorff_distance(pos_d, pos_gamma)
        e_gamma_des2[i] = np.mean(np.linalg.norm(pos_d - pos_gamma, axis=1))

        # Find the global max error across all arrays
    max_error = max(np.max(e_save), np.max(e_save2),
                    np.max(e_est_save), np.max(e_est_save2),
                    np.max(e_real_save), np.max(e_real_save2),
                    np.max(e_gamma_beta), np.max(e_gamma_beta2),
                    np.max(e_gamma_des),np.max(e_gamma_des2))

    # Add a 10% margin
    y_limit = max_error * 1.1

    # Create a figure with 3 subplots (vertically stacked)
    fig, axes = plt.subplots(2, 3, figsize=(10, 12), sharey=True)

    # Plot 1: Approximation Error
    axes[0, 0].plot(t, e_gamma_beta.flatten(), linewidth=2.0, label='AHD')
    axes[0, 0].plot(t, e_gamma_beta2.flatten(), linewidth=2.0, label='ED')
    axes[0, 0].set_ylabel("Error (m)")
    axes[0, 0].set_title("Approximation Error: AHD(x, x_gamma)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Visible Error
    axes[1,0].plot(t, e_save.flatten(), linewidth=2.0, label='AHD')
    axes[1,0].plot(t, e_save2.flatten(), linewidth=2.0, label='ED')
    axes[1,0].set_ylabel("Error (m)")
    axes[1,0].set_title("Visible Error: AHD(x_e, x_d)")
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Plot 3: Estimation Error
    axes[0,1].plot(t, e_est_save.flatten(), linewidth=2.0, label='AHD')
    axes[0,1].plot(t, e_est_save2.flatten(), linewidth=2.0, label='ED')
    axes[0,1].set_ylabel("Error (m)")
    axes[0,1].set_title("Estimation Error: AHD(x, x_e)")
    axes[0,1].legend()
    axes[0,1].grid(True)

    # Plot 4: Real Error
    axes[1,1].plot(t, e_real_save.flatten(), linewidth=2.0, label='AHD')
    axes[1,1].plot(t, e_real_save2.flatten(), linewidth=2.0, label='ED')
    axes[1,1].set_xlabel("Time (s)")
    axes[1,1].set_ylabel("Error (m)")
    axes[1,1].set_title("Real Error: AHD(x, x_d)")
    axes[1,1].legend()
    axes[1,1].grid(True)

    # Plot 4: Real Error
    axes[1, 2].plot(t, e_gamma_des.flatten(), linewidth=2.0, label='AHD')
    axes[1, 2].plot(t, e_gamma_des2.flatten(), linewidth=2.0, label='ED')
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Error (m)")
    axes[1, 2].set_title("Planner Error: AHD(x_gamma, x_d)")
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Set the same y-axis range for all subplots
    for i in range(2):
        for j in range(2):
            axes[i,j].set_ylim(0, y_limit)

    # Adjust layout for better spacing
    plt.tight_layout()

def points_coord_estimator(quad_positions, rows, cols):
    quad_positions2 = [[rows, cols], [rows, 1], [1, 1], [1, cols]]
    for quad_pos in quad_positions:
        if quad_pos not in quad_positions2:
            quad_positions2.append(quad_pos)
    x_actuators2 = (np.array(quad_positions2))
    points_coord2 = x_actuators2 - 1
    quad_indices2 = func_quad_indices(quad_positions2, cols)
    return [points_coord2, quad_indices2]


def plot_forces(t, u_save):
    """
    Plots the forces over time as three subplots for x, y, and z components.

    Parameters:
    u_save (numpy array): A 2D array of shape (3 * n_actuators, iter),
                          where each column represents the forces at a given time step.
    """
    if u_save.ndim != 2:
        raise ValueError("Input u_save must be a 2D array of shape (3 * n_actuators, iter).")

    n_actuators = u_save.shape[0] // 3  # Number of actuators
    iter_count = u_save.shape[1]  # Number of time steps
    time_steps = np.arange(iter_count)

    # Extract force components
    forces_x = u_save[0::3, :]
    forces_y = u_save[1::3, :]
    forces_z = u_save[2::3, :]

    # Plot forces over time
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i in range(n_actuators):
        axes[0].plot(t, forces_x[i, :], label=f"Actuator {i + 1}")
        axes[1].plot(t, forces_y[i, :], label=f"Actuator {i + 1}")
        axes[2].plot(t, forces_z[i, :], label=f"Actuator {i + 1}")

    # Labels and legends
    axes[0].set_ylabel("Force X")
    axes[0].set_title("Forces in X-direction")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_ylabel("Force Y")
    axes[1].set_title("Forces in Y-direction")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_ylabel("Force Z")
    axes[2].set_title("Forces in Z-direction")
    axes[2].legend()
    axes[2].set_xlabel("Time step")
    axes[2].grid(True)

    plt.tight_layout()

def plot_positions(t, x_save, xd_save):
    cmap = plt.get_cmap("tab10")

    iter_count = x_save.shape[1]  # Number of time steps
    time_steps = np.arange(iter_count)

    # Extract force components
    positions_x = x_save[0::6, :]
    positions_y = x_save[1::6, :]
    positions_z = x_save[2::6, :]

    positions_d_x = xd_save[0::6, :]
    positions_d_y = xd_save[1::6, :]
    positions_d_z = xd_save[2::6, :]

    # Plot forces over time
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i in range(4):
        axes[0].plot(t, positions_x[i, :], color=cmap(0), label=f"x_{i + 1}")
        axes[0].plot(t, positions_d_x[i, :], '--', color=cmap(0),  label=f"xd_{i + 1}",alpha=0.3)
        axes[1].plot(t, positions_y[i, :], color=cmap(1), label=f"y_{i + 1}")
        axes[1].plot(t, positions_d_y[i, :], '--', color=cmap(1), label=f"yd_{i + 1}", alpha=0.3)
        axes[2].plot(t, positions_z[i, :], color=cmap(2), label=f"z_{i + 1}")
        axes[2].plot(t, positions_d_z[i, :], '--', color=cmap(2), label=f"zd_{i + 1}", alpha=0.3)

    # Labels and legends
    axes[0].set_ylabel("Position X")
    axes[0].set_title("Positions in X-direction")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].set_ylabel("Position Y")
    axes[1].set_title("Positions in Y-direction")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].set_ylabel("Position Z")
    axes[2].set_title("Positions in Z-direction")
    axes[2].legend()
    axes[2].set_xlabel("Time step")
    axes[2].grid(True)

    plt.tight_layout()


def plot_u_errors(t, u_components_save):
    u_H = u_components_save[0, :]
    u_G = u_components_save[1, :]
    u_0 = u_components_save[2, :]
    u_H_b = u_components_save[3, :]

    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))

    signals = [u_H, u_G, u_0, u_H_b]
    titles = ["u_H", "u_G", "u_0", "u_H_b"]

    # Find max absolute value for each signal and set ylim
    for ax, signal, title in zip(axes.flat, signals, titles):
        ax.plot(t, signal.flatten(), linewidth=2.0)
        ax.set_ylabel("Signal")
        ax.set_title(title)
        ax.grid(True)

        max_val = np.max(signal) * 1.1
        ax.set_ylim(0, max_val)

    # Adjust layout for better spacing
    plt.tight_layout()