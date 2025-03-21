import numpy as np
import do_mpc
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy import signal
import casadi as ca
from util import *

def norm_matrix(i1, j1, i2, j2, x, n_points):
    aux = x[6 * n_points * (j1 - 1) + 6 * i1 - 6:6 * n_points * (j1 - 1) + 6 * i1 - 3]
    aux2 = x[6 * n_points * (j2 - 1) + 6 * i2 - 6:6 * n_points * (j2 - 1) + 6 * i2 - 3]
    return np.linalg.norm(aux - aux2)

def norm_matrix2(i1, j1, i2, j2, x, n_points):
    aux = x[6 * n_points * (j1 - 1) + 6 * i1 - 6:6 * n_points * (j1 - 1) + 6 * i1 - 3]
    aux2 = x[6 * n_points * (j2 - 1) + 6 * i2 - 6:6 * n_points * (j2 - 1) + 6 * i2 - 3]
    diff = aux-aux2
    squared = diff ** 2
    sum_squared = squared[0] + squared[1] + squared[2]
    euclidean_norm = sum_squared ** 0.5
    return euclidean_norm

def x_dot(i1, j1, i2, j2, x, n_points):
    aux = x[6 * n_points * (j1 - 1) + 6 * i1 - 6:6 * n_points * (j1 - 1) + 6 * i1 - 3]
    aux2 = x[6 * n_points * (j2 - 1) + 6 * i2 - 6:6 * n_points * (j2 - 1) + 6 * i2 - 3]
    return np.outer(aux - aux2, aux - aux2)


def component_A(i1, j1, i2, j2, x, n_points, k, l0):
    norm_val = norm_matrix(i1, j1, i2, j2, x, n_points)
    x_dot_val = x_dot(i1, j1, i2, j2, x, n_points)
    return (k * l0 / norm_val * np.eye(3)) - (k * l0 * x_dot_val / norm_val ** 3)


def A_linearized(x, k, k2, k3, c1, c2, m, l0, n_points, n_points2):
    A = np.zeros((n_points * n_points2 * 6, n_points * n_points2 * 6))

    for j in range(1, n_points2 + 1):
        for i in range(1, n_points + 1):
            pos_x = slice(6 * n_points * (j - 1) + 6 * (i - 1), 6 * n_points * (j - 1) + 6 * (i - 1) + 3)
            pos_v = slice(6 * n_points * (j - 1) + 6 * (i - 1) + 3, 6 * n_points * (j - 1) + 6 * (i - 1) + 6)
            A[pos_x, pos_v] = np.eye(3)

            c = c2 if (i == 1 and j == 1) or (i == 1 and j == n_points2) or (i == n_points and j == 1) or (
                        i == n_points and j == n_points2) else c1

            A[pos_v, pos_v] = -c / m[i - 1, j - 1] * np.eye(3)

            aux_same = np.zeros((3, 3))

            if i < n_points:
                aux_same = aux_same - k * np.eye(3) + component_A(i, j, i + 1, j, x, n_points, k, l0)
                A[pos_v, pos_x.start + 6:pos_x.start + 9] = k * np.eye(3) - component_A(i, j, i + 1, j, x, n_points, k,
                                                                                        l0)

            if i < n_points - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * component_A(i, j, i + 2, j, x, n_points, k3, l0)
                A[pos_v, pos_x.start + 12:pos_x.start + 15] = k3 * np.eye(3) - 2 * component_A(i, j, i + 2, j, x,
                                                                                               n_points, k3, l0)

            if i > 1:
                aux_same = aux_same - k * np.eye(3) + component_A(i, j, i - 1, j, x, n_points, k, l0)
                A[pos_v, pos_x.start - 6:pos_x.start - 3] = k * np.eye(3) - component_A(i, j, i - 1, j, x, n_points, k,
                                                                                        l0)

            if i > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * component_A(i, j, i - 2, j, x, n_points, k3, l0)
                A[pos_v, pos_x.start - 12:pos_x.start - 9] = k3 * np.eye(3) - 2 * component_A(i, j, i - 2, j, x,
                                                                                              n_points, k3, l0)

            if j < n_points2:
                aux_same = aux_same - k * np.eye(3) + component_A(i, j, i, j + 1, x, n_points, k, l0)
                A[pos_v, pos_x.start + 6 * n_points:pos_x.start + 6 * n_points + 3] = k * np.eye(3) - component_A(i, j,
                                                                                                                  i,
                                                                                                                  j + 1,
                                                                                                                  x,
                                                                                                                  n_points,
                                                                                                                  k, l0)

            if j < n_points2 - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * component_A(i, j, i, j + 2, x, n_points, k3, l0)
                A[pos_v, pos_x.start + 12 * n_points:pos_x.start + 12 * n_points + 3] = k3 * np.eye(
                    3) - 2 * component_A(i, j, i, j + 2, x, n_points, k3, l0)

            if j > 1:
                aux_same = aux_same - k * np.eye(3) + component_A(i, j, i, j - 1, x, n_points, k, l0)
                A[pos_v, pos_x.start - 6 * n_points:pos_x.start - 6 * n_points + 3] = k * np.eye(3) - component_A(i, j,
                                                                                                                  i,
                                                                                                                  j - 1,
                                                                                                                  x,
                                                                                                                  n_points,
                                                                                                                  k, l0)

            if j > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * component_A(i, j, i, j - 2, x, n_points, k3, l0)
                A[pos_v, pos_x.start - 12 * n_points:pos_x.start - 12 * n_points + 3] = k3 * np.eye(
                    3) - 2 * component_A(i, j, i, j - 2, x, n_points, k3, l0)

            if i < n_points and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + component_A(i, j, i + 1, j + 1, x, n_points, k2, np.sqrt(2) * l0)
                A[pos_v, pos_x.start + 6 * n_points + 6:pos_x.start + 6 * n_points + 9] = k2 * np.eye(3) - component_A(
                    i, j, i + 1, j + 1, x, n_points, k2, np.sqrt(2) * l0)

            if i > 1 and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + component_A(i, j, i - 1, j - 1, x, n_points, k2, np.sqrt(2) * l0)
                A[pos_v, pos_x.start - 6 * n_points - 6:pos_x.start - 6 * n_points - 3] = k2 * np.eye(3) - component_A(
                    i, j, i - 1, j - 1, x, n_points, k2, np.sqrt(2) * l0)

            if i < n_points and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + component_A(i, j, i + 1, j - 1, x, n_points, k2, np.sqrt(2) * l0)
                A[pos_v, pos_x.start - 6 * n_points + 6:pos_x.start - 6 * n_points + 9] = k2 * np.eye(3) - component_A(
                    i, j, i + 1, j - 1, x, n_points, k2, np.sqrt(2) * l0)

            if i > 1 and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + component_A(i, j, i - 1, j + 1, x, n_points, k2, np.sqrt(2) * l0)
                A[pos_v, pos_x.start + 6 * n_points - 6:pos_x.start + 6 * n_points - 3] = k2 * np.eye(3) - component_A(
                    i, j, i - 1, j + 1, x, n_points, k2, np.sqrt(2) * l0)

            A[pos_v, pos_x] = aux_same

    return A


def K_matrix(x, k, k2, k3, c1, c2, l0, n_points, n_points2):
    A = np.zeros((n_points * n_points2 * 6, n_points * n_points2 * 6))

    for j in range(1, n_points2 + 1):
        for i in range(1, n_points + 1):
            pos_x = slice(6 * n_points * (j - 1) + 6 * (i - 1), 6 * n_points * (j - 1) + 6 * (i - 1) + 3)
            pos_v = slice(6 * n_points * (j - 1) + 6 * (i - 1) + 3, 6 * n_points * (j - 1) + 6 * (i - 1) + 6)

            A[pos_x, pos_v] = np.eye(3)

            c = c2 if (i == 1 and j == 1) or (i == 1 and j == n_points2) or (i == n_points and j == 1) or (
                        i == n_points and j == n_points2) else c1
            A[pos_v, pos_v] = -c * np.eye(3)

            aux_same = np.zeros((3, 3))

            if i < n_points:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix(i, j, i + 1, j, x, n_points) * np.eye(3)
                A[pos_v, pos_x.start + 6:pos_x.start + 9] = k * np.eye(3) - k * l0 / norm_matrix(i, j, i + 1, j, x,
                                                                                                 n_points) * np.eye(3)

            if i < n_points - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix(i, j, i + 2, j, x, n_points) * np.eye(
                    3)
                A[pos_v, pos_x.start + 12:pos_x.start + 15] = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix(i, j, i + 2, j,
                                                                                                         x,
                                                                                                         n_points) * np.eye(
                    3)

            if i > 1:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix(i, j, i - 1, j, x, n_points) * np.eye(3)
                A[pos_v, pos_x.start - 6:pos_x.start - 3] = k * np.eye(3) - k * l0 / norm_matrix(i, j, i - 1, j, x,
                                                                                                 n_points) * np.eye(3)

            if i > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix(i, j, i - 2, j, x, n_points) * np.eye(
                    3)
                A[pos_v, pos_x.start - 12:pos_x.start - 9] = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix(i, j, i - 2, j,
                                                                                                        x,
                                                                                                        n_points) * np.eye(
                    3)

            if j < n_points2:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix(i, j, i, j + 1, x, n_points) * np.eye(3)
                A[pos_v, pos_x.start + 6 * n_points:pos_x.start + 6 * n_points + 3] = k * np.eye(
                    3) - k * l0 / norm_matrix(i, j, i, j + 1, x, n_points) * np.eye(3)

            if j < n_points2 - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix(i, j, i, j + 2, x, n_points) * np.eye(
                    3)
                A[pos_v, pos_x.start + 12 * n_points:pos_x.start + 12 * n_points + 3] = k3 * np.eye(
                    3) - 2 * k3 * l0 / norm_matrix(i, j, i, j + 2, x, n_points) * np.eye(3)

            if j > 1:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix(i, j, i, j - 1, x, n_points) * np.eye(3)
                A[pos_v, pos_x.start - 6 * n_points:pos_x.start - 6 * n_points + 3] = k * np.eye(
                    3) - k * l0 / norm_matrix(i, j, i, j - 1, x, n_points) * np.eye(3)

            if j > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix(i, j, i, j - 2, x, n_points) * np.eye(
                    3)
                A[pos_v, pos_x.start - 12 * n_points:pos_x.start - 12 * n_points + 3] = k3 * np.eye(
                    3) - 2 * k3 * l0 / norm_matrix(i, j, i, j - 2, x, n_points) * np.eye(3)

            if i < n_points and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix(i, j, i + 1, j + 1, x,
                                                                                          n_points) * np.eye(3)
                A[pos_v, pos_x.start + 6 * n_points + 6:pos_x.start + 6 * n_points + 9] = k2 * np.eye(3) - k2 * np.sqrt(
                    2) * l0 / norm_matrix(i, j, i + 1, j + 1, x, n_points) * np.eye(3)

            if i > 1 and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix(i, j, i - 1, j - 1, x,
                                                                                          n_points) * np.eye(3)
                A[pos_v, pos_x.start - 6 * n_points - 6:pos_x.start - 6 * n_points - 3] = k2 * np.eye(3) - k2 * np.sqrt(
                    2) * l0 / norm_matrix(i, j, i - 1, j - 1, x, n_points) * np.eye(3)

            if i < n_points and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix(i, j, i + 1, j - 1, x,
                                                                                          n_points) * np.eye(3)
                A[pos_v, pos_x.start - 6 * n_points + 6:pos_x.start - 6 * n_points + 9] = k2 * np.eye(3) - k2 * np.sqrt(
                    2) * l0 / norm_matrix(i, j, i + 1, j - 1, x, n_points) * np.eye(3)

            if i > 1 and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix(i, j, i - 1, j + 1, x,
                                                                                          n_points) * np.eye(3)
                A[pos_v, pos_x.start + 6 * n_points - 6:pos_x.start + 6 * n_points - 3] = k2 * np.eye(3) - k2 * np.sqrt(
                    2) * l0 / norm_matrix(i, j, i - 1, j + 1, x, n_points) * np.eye(3)

            A[pos_v, pos_x] = aux_same

    return A


def K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2):
    A = np.zeros((n_points * n_points2 * 6, n_points * n_points2 * 6))

    for j in range(1, n_points2 + 1):
        for i in range(1, n_points + 1):
            pos_x1 = 6 * n_points * (j - 1) + 6 * (i - 1)
            pos_x2 = 6 * n_points * (j - 1) + 6 * (i - 1) + 3
            pos_v1 = 6 * n_points * (j - 1) + 6 * (i - 1) + 3
            pos_v2 = 6 * n_points * (j - 1) + 6 * (i - 1) + 6

            A[pos_x1: pos_x2, pos_v1: pos_v2] = np.eye(3)

            c = c2 if (i == 1 and j == 1) or (i == 1 and j == n_points2) or (i == n_points and j == 1) or (
                        i == n_points and j == n_points2) else c1
            A[pos_v1: pos_v2, pos_v1: pos_v2] = -c * np.eye(3)

            aux_same = np.zeros((3, 3))

            if i < n_points:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix2(i, j, i + 1, j, x, n_points) * np.eye(3)
                auxiliar_value = k * np.eye(3) - k * l0 / norm_matrix2(i, j, i + 1, j, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1+ 6+bb] = auxiliar_value[aa, bb]

            if i < n_points - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix2(i, j, i + 2, j, x, n_points)*np.eye(3)
                auxiliar_value = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix2(i, j, i + 2, j, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1+ 12+bb] = auxiliar_value[aa, bb]

            if i > 1:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix2(i, j, i - 1, j, x, n_points) * np.eye(3)
                auxiliar_value= k * np.eye(3) - k * l0 / norm_matrix2(i, j, i - 1, j, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1-6+bb] = auxiliar_value[aa, bb]


            if i > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix2(i, j, i - 2, j, x, n_points)*np.eye(3)
                auxiliar_value = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix2(i, j, i - 2, j, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1-12+bb] = auxiliar_value[aa, bb]

            if j < n_points2:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix2(i, j, i, j + 1, x, n_points) * np.eye(3)
                auxiliar_value = k * np.eye(3) - k * l0 / norm_matrix2(i, j, i, j + 1, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1+ 6*n_points+bb] = auxiliar_value[aa, bb]

            if j < n_points2 - 1:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix2(i, j, i, j + 2, x, n_points)*np.eye(3)
                auxiliar_value = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix2(i, j, i, j + 2, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1+ 12*n_points+bb] = auxiliar_value[aa, bb]

            if j > 1:
                aux_same = aux_same - k * np.eye(3) + k * l0 / norm_matrix2(i, j, i, j - 1, x, n_points) * np.eye(3)
                auxiliar_value = k * np.eye(3) - k * l0 / norm_matrix2(i, j, i, j - 1, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 - 6*n_points+bb] = auxiliar_value[aa, bb]

            if j > 2:
                aux_same = aux_same - k3 * np.eye(3) + 2 * k3 * l0 / norm_matrix2(i, j, i, j - 2, x, n_points)*np.eye(3)
                auxiliar_value = k3 * np.eye(3) - 2 * k3 * l0 / norm_matrix2(i, j, i, j - 2, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 - 12 * n_points + bb] = auxiliar_value[aa, bb]

            if i < n_points and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i + 1, j + 1, x,
                                                                                          n_points) * np.eye(3)
                auxiliar_value= k2 * np.eye(3) - k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i + 1, j + 1, x, n_points)*np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 + 6* n_points + 6+ bb] = auxiliar_value[aa, bb]

            if i > 1 and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i - 1, j - 1, x,
                                                                                          n_points) * np.eye(3)
                auxiliar_value= k2 * np.eye(3) - k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i - 1, j - 1, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 - 6* n_points - 6+ bb] = auxiliar_value[aa, bb]

            if i < n_points and j > 1:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i + 1, j - 1, x,
                                                                                          n_points) * np.eye(3)
                auxiliar_value = k2 * np.eye(3) - k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i + 1, j - 1, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 - 6 * n_points + 6 + bb] = auxiliar_value[aa, bb]

            if i > 1 and j < n_points2:
                aux_same = aux_same - k2 * np.eye(3) + k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i - 1, j + 1, x,
                                                                                          n_points) * np.eye(3)
                auxiliar_value = k2 * np.eye(3) - k2 * np.sqrt(2) * l0 / norm_matrix2(i, j, i - 1, j + 1, x, n_points) * np.eye(3)
                for aa in range(3):
                    for bb in range(3):
                        A[pos_v1 + aa, pos_x1 + 6 * n_points - 6 + bb] = auxiliar_value[aa, bb]

            for aa in range(3):
                for bb in range(3):
                    A[pos_v1, pos_x1] = aux_same[aa, bb]

    A = np.nan_to_num(A, nan=0.0)
    return A


def k_lqr(n_points, n_points2, k, k2, k3, c1, c2, l0, m, m_uav, x_actuators, x, Q, R):
    n_actuators = x_actuators.shape[0]
    x_actuators_2 = np.zeros((n_actuators, 3))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    # Linearized system
    T0 = 1
    theta0 = 0.0
    phi0 = 0.0

    A = A_linearized(x * 1.02, k, k2, k3, c1, c2, m, l0, n_points, n_points2)
    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                    x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Control parameters

    '''
    print('A=',A)
    print('B=',B)
    print('Q=',Q)
    print('R=',R)
    '''
    # LQR Calculation
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    K = np.round(K, 6)

    return K


def k_dlqr(n_points, n_points2, k, k2, k3, c1, c2, l0, m, m_uav, x_actuators, x, Q, R, n_visible_points,
           x_visible_points_2, delta):
    n_actuators = x_actuators.shape[0]
    x_actuators_2 = np.zeros((n_actuators, 3))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    # Linearized system
    T0 = 1
    theta0 = 0.0
    phi0 = 0.0

    A = A_linearized(x * 1.02, k, k2, k3, c1, c2, m, l0, n_points, n_points2)
    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                    x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Control parameters

    C = np.zeros((6 * n_visible_points, 6 * n_points * n_points2))
    print(x_visible_points_2)
    for i in range(n_visible_points):
        start_col = int(x_visible_points_2[i, 0])
        C[6 * i:6 * (i + 1), start_col:start_col + 6] = np.eye(6)

    D = np.zeros((6 * n_visible_points, 3 * n_actuators))

    sys_continuous = signal.StateSpace(A, B, C, D)
    sys_discrete = sys_continuous.to_discrete(delta, method='zoh')

    # Extract the discrete-time system matrices A_d and B_d
    A_d, B_d = sys_discrete.A, sys_discrete.B

    # LQR Calculation
    P = solve_discrete_are(A_d, B_d, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
    # Round the gain matrix to 6 decimal places
    K = np.round(K, 6)

    return K


def k_dlqr_V2(n_points, n_points2, k, k2, k3, c1, c2, l0, mass_points, m_uav, x_actuators, x, Q_vector, R_vector,
              delta):
    n_actuators = x_actuators.shape[0]
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    A = A_linearized(x * 1.02, k, k2, k3, c1, c2, m, l0, n_points, n_points2)
    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3)
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                    x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Control parameters

    C = np.zeros((6 * n_visible_points, 6 * n_points * n_points2))
    D = np.zeros((6 * n_visible_points, 3 * n_actuators))

    sys_continuous = signal.StateSpace(A, B, C, D)
    sys_discrete = sys_continuous.to_discrete(delta, method='zoh')

    # Extract the discrete-time system matrices A_d and B_d
    A_d, B_d = sys_discrete.A, sys_discrete.B

    Q = Q_vector[4] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[3] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # LQR Calculation
    P = solve_discrete_are(A_d, B_d, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
    # Round the gain matrix to 6 decimal places
    K = np.round(K, 6)

    return K


def init_MPC_model(x, k, k2, k3, c1, c2, l0, n_points, n_points2, n_actuators, x_actuators, mass_points, m_uav,
                   Q_vector, R_vector, delta, u_limits):
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    A = A_linearized(x * 1.01, k, k2, k3, c1, c2, m, l0, n_points, n_points2)
    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Control parameters
    C = np.zeros((6 * n_visible_points, 6 * n_points * n_points2))
    D = np.zeros((6 * n_visible_points, 3 * n_actuators))

    sys_continuous = signal.StateSpace(A, B, C, D)
    sys_discrete = sys_continuous.to_discrete(delta, method='zoh')

    # Extract the discrete-time system matrices A_d and B_d
    A_d, B_d = sys_discrete.A, sys_discrete.B

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_p', var_name='xref', shape=(6 * n_points * n_points2, 1))  # desired state

    x_next = A_d @ x + B_d @ u
    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 10,  # Prediction horizon
        't_step': delta,
        'state_discretization': 'discrete',
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    Q = Q_vector[3] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = (x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]), n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]), n_actuators)

    return mpc


def init_MPC_model2(x2, k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            #G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = 20*m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.LinearModel(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_p', var_name='xref', shape=(6 * n_points * n_points2, 1))  # desired state

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix(x2, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ (K_spring @ x ) + B @ u - G)
    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 10,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma57'  # Use a faster linear solver
        }

    }

    mpc.set_param(**setup_mpc)

    Q = Q_vector[3] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = (x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]), n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]), n_actuators)

    return mpc

def init_MPC_model3(x2, k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            #G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = 20*m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_p', var_name='xref', shape=(6 * n_points * n_points2, 1))  # desired state

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ (K_spring @ x ) + B @ u - G)
    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 10,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
        }

    }

    mpc.set_param(**setup_mpc)

    Q = Q_vector[3] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = (x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]), n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]), n_actuators)

    return mpc

def init_MPC_model4(x2, k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, xd_save, N_horizon):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            #G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.LinearModel(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(6 * n_points * n_points2, 1))
    #K_spring = model.set_variable(var_type='_p', var_name='K_spring', shape=(6 * n_points * n_points2, 6*n_points*n_points2))  # Matrix_K

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix(x2, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ (K_spring @ x ) + B @ u - G)
    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma57'  # Use a faster linear solver
        }

    }

    mpc.set_param(**setup_mpc)

    Q = Q_vector[3] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = (x - x_ref).T @ Q @ (x - x_ref) # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]), n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]), n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'x_ref'] =  xd_save[:, int(t_now/mpc_dt + k)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc

def init_MPC_model5(x2, k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, xd_save, N_horizon):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            # G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(6 * n_points * n_points2, 1))


    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    Q = Q_vector[3] * np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = (x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref) + u.T @ R @ u  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'x_ref'] = xd_save[:, int(t_now / mpc_dt + k)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc


def init_MPC_model6(x2, k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, xd_save, N_horizon):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            # G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(6 * n_points * n_points2, 1))


    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    Q = np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
        Q[6 * yu - 1, 6 * yu - 1] = Q_vector[3]   # velocity in z
    for i in range(n_actuators):
        index_actuation = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0])
        Q[index_actuation - 6:index_actuation - 4, index_actuation - 6:index_actuation - 4] = Q_vector[5] * np.eye(2)  # x and y of quads
        Q[index_actuation - 4, index_actuation - 4] = Q_vector[5]  # Altitude z of quad
        Q[index_actuation - 3:index_actuation - 1, index_actuation - 3:index_actuation - 1] = Q_vector[6] * np.eye(2)  # velocity in x and y of quads
        Q[index_actuation - 1, index_actuation - 1] = Q_vector[7]  # velocity in z

    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = 0*(x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref)  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=R_vector[0])  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'x_ref'] = xd_save[:, int(t_now / mpc_dt + k)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc


def init_MPC_model7(k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, xd_save, N_horizon):
    mpc_dt = delta
    n_visible_points = n_actuators
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            # G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref = model.set_variable(var_type='_tvp', var_name='x_ref', shape=(6 * n_points * n_points2, 1))


    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    Q = np.eye(6 * n_points * n_points2)  # velocity in z
    for yu in range(1, n_points * n_points2 + 1):
        Q[6 * yu - 6:6 * yu - 4, 6 * yu - 6:6 * yu - 4] = Q_vector[0] * np.eye(2)  # x and y
        Q[6 * yu - 4, 6 * yu - 4] = Q_vector[1]  # Altitude z
        Q[6 * yu - 3:6 * yu - 1, 6 * yu - 3:6 * yu - 1] = Q_vector[2] * np.eye(2)  # velocity in x and y
        Q[6 * yu - 1, 6 * yu - 1] = Q_vector[3]   # velocity in z
    for i in range(n_actuators):
        index_actuation = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0])
        Q[index_actuation - 6:index_actuation - 4, index_actuation - 6:index_actuation - 4] = Q_vector[5] * np.eye(2)  # x and y of quads
        Q[index_actuation - 4, index_actuation - 4] = Q_vector[5]  # Altitude z of quad
        Q[index_actuation - 3:index_actuation - 1, index_actuation - 3:index_actuation - 1] = Q_vector[6] * np.eye(2)  # velocity in x and y of quads
        Q[index_actuation - 1, index_actuation - 1] = Q_vector[7]  # velocity in z

    R = R_vector[1] * np.eye(3 * n_actuators)  # force in z
    for yi in range(n_actuators):
        R[3 * yi + 1:3 * yi + 3, 3 * yi + 1:3 * yi + 3] = R_vector[0] * np.eye(2)  # force in x and y

    # Stage cost: (x - x_ref)^2 + lambda * u^2
    mterm = 0*(x - x_ref).T @ Q @ (x - x_ref)  # Terminal cost (optional)
    lterm = (x - x_ref).T @ Q @ (x - x_ref)  # Stage cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=R_vector[0])  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'x_ref'] = xd_save[:, int(t_now / mpc_dt + k)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc


def init_MPC_0(k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, xd_0_save, N_horizon):
    mpc_dt = 20*delta
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            #G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    x_ref_0 = model.set_variable(var_type='_tvp', var_name='x_ref_0', shape=(6, 1))


    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    x_3 = ca.reshape(x, (6, n_points*n_points2))/(n_points * n_points2)

    one_1 = np.ones((n_points*n_points2,1))# Define an n x 1 column vector

    x_0 = x_3 @ one_1

    Q = 100000*np.eye(6)  # velocity in z
    #Q = 10 * np.eye(6)  # velocity in z

    Q[3:5, 3:5] = 5 * np.eye(2)

    Q[5, 5] = 5

    lterm = (x_0 - x_ref_0).T @ Q @ (x_0 - x_ref_0)  # Stage cost

    mterm = 0*lterm  # Terminal cost (optional)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'x_ref_0'] = xd_0_save[:, int(t_now /mpc_dt + k -1 )]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc

def init_MPC_Rs(k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, Rs_d_save, shape_save, N_horizon):
    mpc_dt = 20*delta
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            #G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    Hd = model.set_variable(var_type='_tvp', var_name='Hd', shape=(3, 3))
    c_b = model.set_variable(var_type='_tvp', var_name='c_b', shape=(3, n_points * n_points2))


    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    one_1 = np.ones((n_points * n_points2, 1)) / (n_points * n_points2)  # Define an n x 1 column vector
    x_3 = ca.reshape(x, (6, n_points*n_points2))
    x_0 = x_3 @ one_1

    x_b = x_3 - x_0
    x_b = x_b[0:3,:]

    e_Rs = ca.reshape((Hd @ c_b - x_b), (3*n_points*n_points2,1))

    print(e_Rs.shape)

    Q = 20*np.eye(3*n_points*n_points2)  # velocity in z

    lterm = (e_Rs).T @ Q @ (e_Rs)  # Stage cost

    mterm = 0*lterm  # Terminal cost (optional)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'Hd'] = Rs_d_save[:, :, int(t_now /mpc_dt + k-1)]
            tvp_template['_tvp', k, 'c_b'] = shape_save[:, :, int(t_now / mpc_dt + k-1)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc

def init_MPC_shape(k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    Q_vector, R_vector,
                    delta, u_limits, g, Rs_d_save, shape_save, N_horizon):
    mpc_dt = 20*delta
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            #G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    c_b = model.set_variable(var_type='_tvp', var_name='c_b', shape=(3, n_points * n_points2))
    #R_h = model.set_variable(var_type='_p', var_name='R_h', shape=(3, 3))

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    one_1 = np.ones((n_points * n_points2, 1)) / (n_points * n_points2)  # Define an n x 1 column vector
    x_3 = ca.reshape(x, (6, n_points*n_points2))
    x_0 = x_3 @ one_1

    x_b = x_3 - x_0
    x_b = x_b[0:3,:]

    # Compute the scaling factor
    #c_s = ca.trace(c_b @ x_b.T)
    #s_h = ca.trace(x_b.T @ R_h @ c_b) / c_s

    #H = s_h * R_h

    H, R_h_var, s_h = matrix_H_3D_ca2(x_b, c_b)

    alpha_H = 4000.0
    alpha_G = 2000.0

    # Control laws
    e_H = H @ c_b - x_b
    G = x_b @ ca.pinv(c_b)
    e_G = G @ c_b - x_b

    e_H = ca.reshape(e_H, (3 * n_points * n_points2, 1))
    e_G = ca.reshape(e_G, (3 * n_points * n_points2, 1))

    Q = np.eye(3*n_points*n_points2)  # velocity in z

    lterm = alpha_H*(e_H).T @ Q @ (e_H) + alpha_G*(e_G).T @ Q @ (e_G)  # Stage cost

    mterm = 0*lterm  # Terminal cost (optional)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'c_b'] = shape_save[:, :, int(t_now / mpc_dt + k-1)]

        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    #R_h_var = np.zeros([3,3])  # Create a 3x3 identity matrix
    #R_h_var = R_h_var.flatten().tolist()  # Convert to a flat list
    #mpc.set_uncertainty_values(R_h=R_h_var)

    return mpc

def init_MPC_general(k, k2, k3, c1, c2, l0,
                    n_points, n_points2, n_actuators, x_actuators,
                    mass_points, m_uav,
                    delta, u_limits, g, Rs_d_save, shape_save, xd_0_save, N_horizon):
    mpc_dt = delta
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            G[6 * n_points * j + 6 * i + 5] =  mass_points* g
            #G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    Hd = model.set_variable(var_type='_tvp', var_name='Hd', shape=(3, 3))
    c_b = model.set_variable(var_type='_tvp', var_name='c_b', shape=(3, n_points * n_points2))
    x_ref_0 = model.set_variable(var_type='_tvp', var_name='x_ref_0', shape=(6, 1))

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring@ x)) + B @ u - G)


    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    one_1 = np.ones((n_points * n_points2, 1)) / (n_points * n_points2)  # Define an n x 1 column vector
    x_3 = ca.reshape(x, (6, n_points*n_points2))
    x_0 = x_3 @ one_1

    x_b = x_3 - x_0
    x_b = x_b[0:3,:]

    H, R_h_var, s_h = matrix_H_3D_ca2(x_b, c_b)

    alpha_H = 200
    alpha_G = 70
    #alpha_Rs = 5500
    alpha_Rs = 18000

    # Control laws
    e_H = H @ c_b - x_b
    #G = x_b @ (c_b.T @ ca.pinv(c_b @ c_b.T))
    G = x_b @ ca.pinv(c_b)
    a = G @ c_b - x_b

    e_H = ca.reshape(e_H, (3 * n_points * n_points2, 1))
    a = ca.reshape(a, (3 * n_points * n_points2, 1))

    e_Rs = ca.reshape((Hd @ c_b - x_b), (3 * n_points * n_points2, 1))

    Q = np.eye(3*n_points*n_points2)  # velocity in z

    #Q_0 = 250000000*np.eye(6)

    Q_0 = 5000000 * np.eye(6)

    Q_0[3:5, 3:5] = 2 * np.eye(2)

    Q_0[5, 5] = 2

    lterm = alpha_H*(e_H).T @ Q @ (e_H) + alpha_G*(a).T @ Q @ (a) + alpha_Rs*(e_Rs).T @ Q @ (e_Rs) + (x_0 - x_ref_0).T @ Q_0 @ (x_0 - x_ref_0)# Stage cost

    mterm = 0*lterm  # Terminal cost (optional)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    #mpc.set_rterm(u=0.1)  # Regularization
    mpc.set_rterm(u=10)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'c_b'] = shape_save[:, :, int(t_now / mpc_dt + k-1)]
            tvp_template['_tvp', k, 'Hd'] = Rs_d_save[:, :, int(t_now / mpc_dt + k - 1)]
            tvp_template['_tvp', k, 'x_ref_0'] = xd_0_save[:, int(t_now / mpc_dt + k - 1)]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc

def init_MPC_Rs_shape(k, k2, k3, c1, c2, l0,
                     n_points, n_points2, n_actuators, x_actuators,
                     mass_points, m_uav,
                     Q_vector, R_vector,
                     delta, u_limits, g, Rs_d_save, shape_save, xd_0_save, N_horizon):
    mpc_dt = 20 * delta
    x_actuators_2 = np.zeros((n_actuators, 3))
    m = mass_points * np.ones((n_points, n_points2))
    for i in range(n_actuators):
        m[x_actuators[i, 0] - 1, x_actuators[i, 1] - 1] = m_uav
        x_actuators_2[i, 0] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 1
        x_actuators_2[i, 1] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 2
        x_actuators_2[i, 2] = 6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3

    M = np.eye(6 * n_points * n_points2)
    G = np.zeros((6 * n_points * n_points2, 1))
    # Update M and G
    for j in range(n_points2):
        for i in range(n_points):
            pos_v = slice(6 * n_points * j + 6 * i + 3, 6 * n_points * j + 6 * i + 6)
            M[pos_v, pos_v] = m[i, j] * np.eye(3)
            G[6 * n_points * j + 6 * i + 5] = mass_points * g
            # G[6 * n_points * j + 6 * i + 5] = m[i, j] * g
    # print("M:", M)
    # print("G:", G)

    B = np.zeros((n_points * n_points2 * 6, 3 * n_actuators))
    B_matrix_3 = np.eye(3) / m_uav
    for i in range(n_actuators):
        B[6 * n_points * (x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 3:6 * n_points * (
                x_actuators[i, 1] - 1) + 6 * (x_actuators[i, 0] - 1) + 6, 3 * i:3 * (i + 1)] = B_matrix_3

    # Define the system model
    model_type = 'discrete'
    model = do_mpc.model.Model(model_type)

    # Define state (x) and control (u) variables
    x = model.set_variable(var_type='_x', var_name='x', shape=(6 * n_points * n_points2, 1))  # [position, velocity]
    u = model.set_variable(var_type='_u', var_name='u', shape=(3 * n_actuators, 1))  # [force]
    Hd = model.set_variable(var_type='_tvp', var_name='Hd', shape=(3, 3))
    c_b = model.set_variable(var_type='_tvp', var_name='c_b', shape=(3, n_points * n_points2))
    x_ref_0 = model.set_variable(var_type='_tvp', var_name='x_ref_0', shape=(6, 1))

    M_inv = np.linalg.inv(M)

    K_spring = K_matrix2(x, k, k2, k3, c1, c2, l0, n_points, n_points2)

    x_next = x + mpc_dt * (M_inv @ ((K_spring @ x)) + B @ u - G)

    model.set_rhs('x', x_next)

    model.setup()

    # Define MPC Controller
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': N_horizon,  # Prediction horizon
        't_step': mpc_dt,
        'state_discretization': 'discrete',
        'open_loop': 0,  # Closed-loop MPC
        'store_full_solution': False,
        'nlpsol_opts': {
            # 'jit': True,
            'ipopt.tol': 1e-3,
            'ipopt.max_iter': 1000,
            'ipopt.print_level': 0,  # Disable IPOPT printing
            'ipopt.ma57_automatic_scaling': 'no',  # Enable MA57 auto scaling
            'ipopt.sb': 'yes',  # Enable silent barrier mode
            'print_time': 0,  # Disable solver timing information
            'ipopt.linear_solver': 'ma27'  # Use a faster linear solver
        }
    }

    mpc.set_param(**setup_mpc)

    one_1 = np.ones((n_points * n_points2, 1)) / (n_points * n_points2)  # Define an n x 1 column vector
    x_3 = ca.reshape(x, (6, n_points * n_points2))
    x_0 = x_3 @ one_1

    x_b = x_3 - x_0
    x_b = x_b[0:3, :]

    H, R_h_var, s_h = matrix_H_3D_ca2(x_b, c_b)

    alpha_H = 5
    alpha_G = 120
    alpha_Rs = 80

    # Control laws
    e_H = H @ c_b - x_b
    G = x_b @ ca.pinv(c_b)
    e_G = G @ c_b - x_b

    e_H = ca.reshape(e_H, (3 * n_points * n_points2, 1))
    e_G = ca.reshape(e_G, (3 * n_points * n_points2, 1))

    e_Rs = ca.reshape((Hd @ c_b - x_b), (3 * n_points * n_points2, 1))

    Q = np.eye(3 * n_points * n_points2)  # velocity i

    # Q_0[3:5, 3:5] = 2 * np.eye(2)

    # Q_0[5, 5] = 2

    lterm = alpha_H * (e_H).T @ Q @ (e_H) + alpha_G * (e_G).T @ Q @ (e_G) + alpha_Rs * (e_Rs).T @ Q @ (e_Rs)   # Stage cost

    mterm = (e_H).T @ Q @ (e_H)  # Terminal cost (optional)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1)  # Regularization

    # Input constraints
    mpc.bounds['lower', '_u', 'u'] = np.tile(np.array([u_limits[0, 0], u_limits[1, 0], u_limits[2, 0]]),
                                             n_actuators)
    mpc.bounds['upper', '_u', 'u'] = np.tile(np.array([u_limits[0, 1], u_limits[1, 1], u_limits[2, 1]]),
                                             n_actuators)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_now):
        for k in range(N_horizon + 1):
            tvp_template['_tvp', k, 'c_b'] = shape_save[:, :, int(t_now / mpc_dt + k - 1)]
            tvp_template['_tvp', k, 'Hd'] = Rs_d_save[:, :, int(t_now / mpc_dt + k - 1)]
            tvp_template['_tvp', k, 'x_ref_0'] = xd_0_save[:, int(t_now / mpc_dt + k - 1)]
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    return mpc