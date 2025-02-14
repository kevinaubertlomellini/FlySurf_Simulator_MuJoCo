import numpy as np
import do_mpc
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy import signal


def norm_matrix(i1, j1, i2, j2, x, n_points):
    aux = x[6 * n_points * (j1 - 1) + 6 * i1 - 6:6 * n_points * (j1 - 1) + 6 * i1 - 3]
    aux2 = x[6 * n_points * (j2 - 1) + 6 * i2 - 6:6 * n_points * (j2 - 1) + 6 * i2 - 3]
    return np.linalg.norm(aux - aux2)


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
