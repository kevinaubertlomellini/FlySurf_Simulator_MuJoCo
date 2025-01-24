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

def shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, x, n_points, c, R_d, s_d, c_0):
    x = np.reshape(x, (3, n_points))
    c = np.reshape(c, (3, n_points))

    c_0 = c_0[:, np.newaxis]

    # Compute centroid of x
    x_0 = np.mean(x, axis=1).reshape(-1, 1)
    x_b = x - np.tile(x_0, n_points)  # Shape of x (x without centroid)

    # Compute centroid of c
    c_00 = np.mean(c, axis=1).reshape(-1, 1)
    c_b = c - np.tile(c_00, n_points)

    # Compute H, R_h, and s_h using a helper function (needs to be implemented)
    H, R_h, s_h = matrix_H_3D(x_b, c_b)
    u_H = H @ c_b - x_b

    # Compute G
    G = x_b @ np.linalg.pinv(c_b)
    u_G = G @ c_b - x_b

    # Compute individual components of u
    u_y = alpha_H * u_H + alpha_G * u_G
    u_0 = -alpha_0 * (x_0 - c_0) @ np.ones((1, n_points))
    Hd = s_d * R_d
    u_Hd = alpha_Hd * (Hd @ c_b - x_b)

    # Combine all components of u
    u = u_y + u_0 + u_Hd
    u = u.flatten()  # Reshape into a 1D array

    return u


def matrix_H_3D(Q_b, C_b):
    # Covariance matrix
    H_R = C_b @ Q_b.T

    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H_R)
    V = Vt.T
    R_h = V @ U.T

    # Scale factor computation
    c_s = np.trace(C_b @ C_b.T)
    s_h = np.trace(Q_b.T @ R_h @ C_b) / c_s

    # Compute H
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
