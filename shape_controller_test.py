import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util import *

# Initialization
x = np.array([-1, 0, 0, 0, 1, 1, 1, 0, 0, 0, -1, 1], dtype=float).reshape(-1, 1)

n_points = 4

# Simulation parameters
steps_change = 400
iter = steps_change*1
delta = 0.025

c = np.array([-2, 0, 0, 0, 1, 0, 2, 0, 0, 0, -1, 0], dtype=float).reshape(-1, 1)
R_d = rotation_matrix(0, 0, 0)
s_d = 1
c_0 = np.array([0, 0, 0], dtype=float).reshape(-1, 1)

alpha_H = 2
alpha_G = 20
alpha_0 = 2
alpha_Hd = 5

u_save = np.zeros((3 * n_points, iter))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Simulation loop
for step in range(iter):
    # Step
    ax.clear()
    u_shape = shape_controller_3D(alpha_H, alpha_G, alpha_0, alpha_Hd, x, n_points, c, R_d, s_d, c_0)
    x = x + u_shape * delta

    x_reshaped = np.reshape(x, (-1, 3))
    u_shape_reshaped = np.reshape(u_shape, (-1, 3))

    # Concatenate the reshaped matrices vertically
    xd = np.hstack((x_reshaped, u_shape_reshaped)).reshape(-1,1)

    u_save[:, step] = u_shape.flatten()

    # Plot
    c2 = c.reshape(3, n_points, order='F')
    c2 = R_d @ (c2 - np.mean(c2, axis=1, keepdims=True))
    c3 = s_d * c2 + c_0

    print(x)
    ax.plot(c3[0, :], c3[1, :], c3[2, :], 'r*', linewidth=0.5)
    ax.plot(x[::3].flatten(), x[1::3].flatten(), x[2::3].flatten(), 'k*', linewidth=2.4)
    ax.plot(c3[0, :], c3[1, :], c3[2, :], 'r-', linewidth=0.5)
    ax.plot([c3[0, 0], c3[0, -1]], [c3[1, 0], c3[1, -1]], [c3[2, 0], c3[2, -1]], 'r-', linewidth=0.5)
    ax.plot(x[::3].flatten(), x[1::3].flatten(), x[2::3].flatten(), 'k-', linewidth=0.5)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.grid(True)
    plt.legend(['Desired positions', 'Actual positions'])
    plt.pause(0.002)

    # Parameter updates
    if step == int(0.2 * steps_change):
        R_d = rotation_matrix(np.pi / 4, np.pi / 4, 0)
    if step == int(0.4 * steps_change):
        c = np.array([-2, 0, 0, 0, 2, 0, 2, 0, 0, 0, -2, 0], dtype=float).reshape(-1, 1)
    if step == int(0.6 * steps_change):
        s_d = 0.6
    if step == int(0.8 * steps_change):
        c_0 = np.array([-1, -1, -0.5], dtype=float).reshape(-1, 1)
    if step == 1:
        plt.pause(5)

t = np.arange(0, iter * delta, delta)

# Plot u_save components
plt.figure()
plt.plot(t, u_save[0, :], linewidth=1.5)
plt.plot(t, u_save[3, :], linewidth=1.5)
plt.plot(t, u_save[6, :], linewidth=1.5)
plt.legend(["ux_1", "ux_2", "ux_3"])
plt.grid(True)


plt.figure()
plt.plot(t, u_save[1, :], linewidth=1.5)
plt.plot(t, u_save[4, :], linewidth=1.5)
plt.plot(t, u_save[7, :], linewidth=1.5)
plt.legend(["uy_1", "uy_2", "uy_3"])
plt.grid(True)
plt.show()
