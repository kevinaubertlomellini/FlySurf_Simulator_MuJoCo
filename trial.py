import do_mpc
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# Define the system model (discrete-time)
model_type = 'discrete'
model = do_mpc.model.Model(model_type)

# Define state (x) and control (u) variables
x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))  # [position, velocity]
u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))  # [force]

# Define `x_des` as a parameter BEFORE setup()
x_des = model.set_variable(var_type='_p', var_name='x_des', shape=(2,1))  # Desired state

g  = model.set_variable(var_type='_p', var_name='g ', shape=(1,1))  # gravity

# System dynamics (discrete-time)
dt = 0.1  # Time step
A = np.array([[1.0, dt], [0, 1.0]])  # State transition
B = np.array([[0], [dt]])  # Control input

x_next = A @ x + B @ (u- g)
model.set_rhs('x', x_next)

model.setup()  # Now we can call setup()

# Define MPC Controller
mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 10,  # Prediction horizon
    't_step': dt,
    'state_discretization': 'discrete',
    'store_full_solution': False,
    'nlpsol_opts': {
                # 'jit': True,
                'ipopt.print_level': 0,  # Disable IPOPT printing
            }
}
mpc.set_param(**setup_mpc)

# Define desired trajectory (example: sinusoidal reference)
def desired_trajectory(t):
    return np.array([[np.sin(0.2 * t)], [0.2 * np.cos(0.2 * t)]])  # [position, velocity]

# Define cost function (tracking error)
Q = np.diag([10.0, 1.0])  # Higher weight on position tracking
R = np.array([[0.01]])  # Control effort penalty

mterm = (x - x_des).T @ Q @ (x - x_des)  # Terminal cost
lterm = (x - x_des).T @ Q @ (x - x_des) + u.T @ R @ u  # Running cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(u=0.01)  # Regularization

# Input constraints
mpc.bounds['lower', '_u', 'u'] = -5.0
mpc.bounds['upper', '_u', 'u'] = 5.0

p_mpc_template = mpc.get_p_template(n_combinations=1)
x_des_value = desired_trajectory(0.0)  # Get desired state at time t
p_mpc_template['_p'] = x_des_value
mpc.set_p_fun(lambda t_now: p_mpc_template)
mpc.setup()

# Initial state
x0 = np.array([[1.0], [0.0]])  # [position=2, velocity=0]
mpc.x0 = x0
mpc.set_initial_guess()

# Simulate MPC control
N_steps = 500
time = np.arange(N_steps) * dt

actual_positions = []
desired_positions = []
actual_velocities = []
desired_velocities = []
control_inputs = []

for t in time:
    x_des_value = desired_trajectory(t)  # Get desired state at time t
    p_mpc_template['_p'] = x_des_value
    mpc.set_p_fun(lambda t_now: p_mpc_template)
    u_mpc = mpc.make_step(x0)  # Compute control input


    # Store data for plotting
    actual_positions.append(x0[0, 0])
    desired_positions.append(x_des_value[0, 0])
    actual_velocities.append(x0[1, 0])
    desired_velocities.append(x_des_value[1, 0])
    control_inputs.append(u_mpc[0, 0])

    # Apply control and update state
    x0 = A @ x0 + B @ u_mpc  # Update state using discrete dynamics

# Plot results
plt.figure(figsize=(10, 5))

# Position tracking plot
plt.subplot(3, 1, 1)
plt.plot(time, desired_positions, 'r--', label="Desired Position")
plt.plot(time, actual_positions, 'b-', label="Actual Position")
plt.xlabel("Time [s]")
plt.ylabel("Position")
plt.legend()
plt.title("MPC Tracking Performance")

plt.subplot(3, 1, 2)
plt.plot(time, desired_velocities, 'r--', label="Desired Velocity")
plt.plot(time, actual_velocities, 'b-', label="Actual Velocity")
plt.xlabel("Time [s]")
plt.ylabel("Velocity")
plt.legend()
plt.title("MPC Tracking Performance 2")

# Control input plot
plt.subplot(3, 1, 3)
plt.plot(time, control_inputs, 'g-', label="Control Input (Force)")
plt.xlabel("Time [s]")
plt.ylabel("Control Input")
plt.legend()
plt.title("Control Effort")

plt.tight_layout()
plt.show()
