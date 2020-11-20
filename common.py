import numpy as np
from scipy.integrate._ivp.ivp import solve_ivp

config = {}
config["car_size"] = 0.1
config["wheel_size"] = (0.075, 0.015)


def draw_initial(axes):
    ax1, ax2 = axes
    plots = {}
    (plots["car"],) = ax1.plot([], [], c="k")
    (plots["car_angle"],) = ax1.plot([], [], c="k")
    (plots["lefT_tire"],) = ax1.plot([], [], c="k")
    (plots["right_tire"],) = ax1.plot([], [], c="k")
    plots["tau0"] = ax2.scatter([], [], c="k")
    plots["tau1"] = ax2.scatter([], [], c="k")
    return plots


def calc_lambda_dot(lambda_, clambda, epsilon):
    return np.atleast_1d(-clambda * (lambda_ - epsilon))


def update_state(fun, t_span, xi):
    sol = solve_ivp(fun, t_span, xi)
    yshape = sol.y.shape
    return sol.y[:, yshape[1] - 1]


def update_lambda(fun, t_span, lambda_):
    sol = solve_ivp(fun, t_span, np.atleast_1d(lambda_),)
    yshape = sol.y.shape
    return sol.y[:, yshape[1] - 1]


# code below comes from PythonLinearNonlinearControl(https://github.com/Shunichi09/PythonLinearNonlinearControl).
# MIT License

# Copyright (c) 2020 Shunichi Sekiguchi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def rotate_pos(pos, angle):
    """ Transformation the coordinate in the angle
    Args:
        pos (numpy.ndarray): local state, shape(data_size, 2)
        angle (float): rotate angle, in radians
    Returns:
        rotated_pos (numpy.ndarray): shape(data_size, 2)
    """
    rot_mat = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )

    return np.dot(pos, rot_mat.T)


def circle(center_x, center_y, radius, start=0.0, end=2 * np.pi, n_point=100):
    """ Create circle matrix
    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        start (float): start angle
        end (float): end angle
    Returns:
        circle x : numpy.ndarray
        circle y : numpy.ndarray
    """
    diff = end - start

    circle_xs = []
    circle_ys = []

    for i in range(n_point + 1):
        circle_xs.append(center_x + radius * np.cos(i * diff / n_point + start))
        circle_ys.append(center_y + radius * np.sin(i * diff / n_point + start))

    return np.array(circle_xs), np.array(circle_ys)


def circle_with_angle(center_x, center_y, radius, angle):
    """ Create circle matrix with angle line matrix
    Args:
        center_x (float): the center x position of the circle
        center_y (float): the center y position of the circle
        radius (float): in meters
        angle (float): in radians
    Returns:
        circle_x (numpy.ndarray): x data of circle
        circle_y (numpy.ndarray): y data of circle
        angle_x (numpy.ndarray): x data of circle angle
        angle_y (numpy.ndarray): y data of circle angle
    """
    circle_x, circle_y = circle(center_x, center_y, radius)

    angle_x = np.array([center_x, center_x + np.cos(angle) * radius])
    angle_y = np.array([center_y, center_y + np.sin(angle) * radius])

    return circle_x, circle_y, angle_x, angle_y


def square(center_x, center_y, shape, angle):
    """ Create square
    Args:
        center_x (float): the center x position of the square
        center_y (float): the center y position of the square
        shape (tuple): the square's shape(width/2, height/2)
        angle (float): in radians
    Returns:
        square_x (numpy.ndarray): shape(5, ), counterclockwise from right-up
        square_y (numpy.ndarray): shape(5, ), counterclockwise from right-up
    """
    # start with the up right points
    # create point in counterclockwise, local
    square_xy = np.array(
        [
            [shape[0], shape[1]],
            [-shape[0], shape[1]],
            [-shape[0], -shape[1]],
            [shape[0], -shape[1]],
            [shape[0], shape[1]],
        ]
    )
    # translate position to world
    # rotation
    trans_points = rotate_pos(square_xy, angle)
    # translation
    trans_points += np.array([center_x, center_y])

    return trans_points[:, 0], trans_points[:, 1]


def plot_car(curr_x):
    """ plot car fucntions
    """
    # cart
    car_x, car_y, car_angle_x, car_angle_y = circle_with_angle(
        curr_x[0], curr_x[1], config["car_size"], curr_x[2]
    )

    # left tire
    center_x = (config["car_size"] + config["wheel_size"][1]) * np.cos(
        curr_x[2] - np.pi / 2.0
    ) + curr_x[0]
    center_y = (config["car_size"] + config["wheel_size"][1]) * np.sin(
        curr_x[2] - np.pi / 2.0
    ) + curr_x[1]

    left_tire_x, left_tire_y = square(
        center_x, center_y, config["wheel_size"], curr_x[2]
    )

    # right tire
    center_x = (config["car_size"] + config["wheel_size"][1]) * np.cos(
        curr_x[2] + np.pi / 2.0
    ) + curr_x[0]
    center_y = (config["car_size"] + config["wheel_size"][1]) * np.sin(
        curr_x[2] + np.pi / 2.0
    ) + curr_x[1]

    right_tire_x, right_tire_y = square(
        center_x, center_y, config["wheel_size"], curr_x[2]
    )

    return (
        car_x,
        car_y,
        car_angle_x,
        car_angle_y,
        left_tire_x,
        left_tire_y,
        right_tire_x,
        right_tire_y,
    )
