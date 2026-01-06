import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.typing import NDArray

# Prepare
# variable s - grid size
# variable x - x-axis points (range from -2 to 2)
# variable y - y-axis points (range from -2 to 2)
# X, Y - grid
# Z = gird of complex numbers


def julia(z, c, max_iter=300) -> NDArray:
    """
    Applies mathematical formula of Julia fractal.
    :param z: Complex numbers grid
    :param c: Complex parameter for Julia fractal
    :param max_iter: Total iterations number
    :return: 2D numpy array that has the same shape as z
    """
    result = np.zeros(z.shape, dtype=int)

    # Copy array to avoid mutations of original one
    zz = z.copy()

    # Iterate max_iters times
    for iteration in range(max_iter):
        # If number exceed 2, apply mask
        escape_mask = np.abs(zz) <= 2

        # Apply fractal formula
        zz[escape_mask] = zz[escape_mask] ** 2 + c
        escaped = escape_mask & (np.abs(zz) > 2)

        result[escaped] = iteration

    return np.clip(result, 0, 255)


def evolve(frame):
    """
    Animation function - changes c over time.
    Point c moves along a curve in space Z
    :param frame:
    :return:
    """
    a, b, d = 3, 2, np.pi / 2
    phi = 4 * np.pi * frame / 720
    re = np.sin(a * phi + d)
    im = np.sin(b * phi)
    c = re + 1j * im

    matrix.set_data(julia(Z, c))
    return [matrix]


if __name__ == "__main__":
    c = np.cos(0) + 1j * np.sin(0)

    # Grid size
    s = 100
    # Array of s elements, each within range -2, 2
    x = np.linspace(-2, 2, s)
    y = np.linspace(-2, 2, s)

    # Initial grid
    X, Y = np.meshgrid(x, y)

    # Complex numbers grid
    Z = X + 1j * Y

    fig, ax = plt.subplots()
    matrix = ax.matshow(julia(Z, c), cmap="magma")
    ani = animation.FuncAnimation(fig, evolve, frames=720, interval=20)
    plt.show()
