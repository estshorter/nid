import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from matplotlib.animation import FuncAnimation

from common import plot_car, update_state, calc_lambda_dot, update_lambda, draw_initial

MAX_STEP = 100
DT = 0.01
TF = 8
clambda = 1
cp = 1.0
cd = 2.0
epsilon = 0.01
LAMBDA_INIT = 0.5


def initialize_xi():
    xi = np.empty(5)
    xi[0] = 2  # posx
    xi[1] = 3  # posy
    xi[2] = np.pi * 7 / 4  # theta
    xi[3] = 0  # foward velocity
    xi[4] = 0  # angular velocity along +Z axis
    return xi


def calc_w_hat_lambda(w, lambda_, lambda_dot):
    whl = np.empty((2, 2))
    whl[0, 0] = 0
    whl[0, 1] = -lambda_ * w
    whl[1, 0] = w / lambda_
    whl[1, 1] = lambda_dot / lambda_
    return whl


def state_equation(xi, u):
    theta = xi[2]
    v1 = xi[3]
    f = np.empty(5)
    f[0] = cos(theta) * v1
    f[1] = sin(theta) * v1
    f[2] = xi[4]  # angular velocity
    f[3:5] = u
    return f


def calc_e_stabilize_law(xi, lambda_, x0):
    x = xi[0:2]
    v = xi[3:5]
    w = xi[4]
    theta = xi[2]
    rot_l_inv = np.empty((2, 2))
    rot_l_inv[0, 0] = cos(theta)
    rot_l_inv[0, 1] = sin(theta)
    rot_l_inv[1, 0] = -sin(theta) / lambda_
    rot_l_inv[1, 1] = cos(theta) / lambda_
    q = x + lambda_ * np.array([cos(theta), sin(theta)])
    lambda_dot = calc_lambda_dot(lambda_, clambda, epsilon)
    e1 = np.zeros(2)
    e1[0] = 1
    whl = calc_w_hat_lambda(w, lambda_, lambda_dot)
    tau = (
        -cp * rot_l_inv @ (q - x0)
        - cd * v
        - cd * lambda_dot * e1
        - whl @ v
        - lambda_dot * (whl - clambda * np.identity(2)) @ e1
    )
    return tau


def update(iter):
    global lambda_, xi
    t = DT * iter
    tau = calc_e_stabilize_law(xi, lambda_, x0)
    xi = update_state(lambda t, xi: state_equation(xi, tau), (t, t + DT), xi)
    lambda_ = update_lambda(
        lambda t, l_: calc_lambda_dot(l_, clambda, epsilon), (t, t + DT), lambda_
    )

    (
        car_x,
        car_y,
        car_angle_x,
        car_angle_y,
        left_tire_x,
        left_tire_y,
        right_tire_x,
        right_tire_y,
    ) = plot_car(xi[0:3])
    plots["car"].set_data(car_x, car_y)
    plots["car_angle"].set_data(car_angle_x, car_angle_y)
    plots["lefT_tire"].set_data(left_tire_x, left_tire_y)
    plots["right_tire"].set_data(right_tire_x, right_tire_y)
    plots["tau0"].set_offsets((t, tau[0]))
    plots["tau1"].set_offsets((t, tau[1]))


xi = initialize_xi()
lambda_ = LAMBDA_INIT

x0 = np.zeros(2)
N = len(np.arange(0, TF, DT))
xi_hist = np.empty((5, N))
tau_hist = np.empty((2, N))
t_hist = np.empty(N)

# save trajectory
for (step, t) in enumerate(np.arange(0, TF, DT)):
    tau = calc_e_stabilize_law(xi, lambda_, x0)
    xi = update_state(lambda t, xi: state_equation(xi, tau), (t, t + DT), xi)
    lambda_ = update_lambda(
        lambda t, l_: calc_lambda_dot(l_, clambda, epsilon), (t, t + DT), lambda_
    )
    xi_hist[:, step] = xi
    tau_hist[:, step] = tau
    t_hist[step] = t

# reset for animation
xi = initialize_xi()
lambda_ = LAMBDA_INIT

fig, axes = plt.subplots(1, 2)
ax1, ax2 = axes

ax1.set_xlim(0, 3.25)
ax1.set_ylim(0, 3.25)
ax1.set_aspect("equal")

plots = draw_initial(axes)
fanm = FuncAnimation(fig, update, interval=10, frames=1000)

ax1.plot(xi_hist[0, :], xi_hist[1, :], zorder=-1)
ax2.plot(t_hist, tau_hist[0, :], t_hist, tau_hist[1, :], zorder=-1)
ax2.legend(["$\dot{v}$", "$\dot{\omega}$"])  # noqa: W605
plt.show()
# fanm.save("movie.mp4", "ffmpeg")
