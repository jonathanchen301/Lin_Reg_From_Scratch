from numpy import *
import matplotlib.pyplot as plt


def compute_y(b, m, x):
    """
    Compute the y given the slope and y-intercept of the line and the x.

    Params:
    b -- y-intercept of the line
    m -- slope of the line
    x -- x of the point
    """

    return m * x + b


def compute_loss(b, m, points, loss_function):
    """
    Compute the distance of all the points to the current line formed by m
    and b and then sum them in order to compute the loss using the loss function.

    Params:
    b -- y-intercept of the line
    m -- slope of the line
    points -- points to compute loss on
    loss_function -- loss function used to compute the loss on (possible: mean_squared_error)
    """

    if loss_function == "mean_squared_error":
        total = 0
        for i in range(len(points)):
            x = points[i][0]
            y = points[i][1]
            total += (y - compute_y(b, m, x)) ** 2
        print("loss:" + str(total / len(points)))
        return total / len(points)
    else:
        raise ValueError("Loss function unrecognized")


def update_weights(b, m, points, alpha, loss_function):
    """
    Update weights based on one iteration through the data based on the gradient
    of b and m

    """
    if loss_function == "mean_squared_error":
        b_gradient = 0
        m_gradient = 0
        for i in range(len(points)):
            x = points[i, 0]
            y = points[i, 1]
            # Compute partial derivative of error function in respect to b and m
            b_gradient += -(2 / len(points)) * (y - compute_y(b, m, x))
            m_gradient += -(2 / len(points)) * (x * (y - compute_y(b, m, x)))
        # Update weights
        b = b - alpha * b_gradient
        m = m - alpha * m_gradient
        return [b, m]
    else:
        raise ValueError("Loss function unrecognized")


def gradient_descent(points, b, m, alpha, epoch, loss_function="mean_squared_error"):
    """
    Update the weights for epoch iterations in order to get to the lowest
    loss.

    Params:
    points -- list of points
    b -- current y-intercept of the line
    m -- current slope of the line
    alpha -- how big of a step to take (default=0.0001)
    epoch -- number of iterations through the data points (default=1000)
    """
    for _ in range(epoch):
        b, m = update_weights(b, m, points, alpha, loss_function)
        print("New b: " + str(b))
        print("New m: " + str(m))
        compute_loss(b, m, points, loss_function)
    return [b, m]


def run():
    # Collect data
    points = genfromtxt(
        "data.csv", delimiter=","
    )  # x is the amount of hours studied and y is the test score

    # Define hyperparameters
    alpha = 0.0001  # Learning rate
    num_epoch = 1000  # Number of iterations through the whole
    loss_function = "mean_squared_error"

    # Initialize weights
    b = 0
    m = 0

    # Train using gradient descent
    final_b, final_m = gradient_descent(points, b, m, alpha, num_epoch, loss_function)
    print("learned_b: " + str(final_b))
    print("learned_m: " + str(final_m))

    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    plt.scatter(x, y)
    plt.plot(
        x, [final_m * xi + final_b for xi in x], color="red", label=f"y = {m}x + {b}"
    )
    plt.show()
