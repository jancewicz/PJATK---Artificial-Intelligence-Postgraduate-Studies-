import numpy as np
from numpy import floating
from numpy.typing import NDArray


def find_values(arr: NDArray, threshold: int):
    """
    Function iterates over given array and find values that are greater than provided threshold. Then average value of
    those numbers are computed, and all indexes of those numbers are saved in list.
    :param arr: 2D Numpy array.
    :param threshold: specified threshold value.
    :return: total count of elements greater than threshold, average number of all elements greater than threshold,
    list of tuples with indexes from 2D array of elements greater than threshold.
    """

    # Declare mask - return True if element is above provided threshold
    above_threshold_mask = arr > threshold

    # Count every element that cross the threshold
    elements_over_threshold_cnt: int = np.count_nonzero(above_threshold_mask)

    average: floating[float] = np.average(arr[above_threshold_mask])
    indexes = [
        (int(row), int(col)) for row, col in zip(*np.where(above_threshold_mask))
    ]

    return elements_over_threshold_cnt, average, indexes


if __name__ == "__main__":
    M = np.arange(1, 101).reshape(10, 10)
    threshold = 75
    nums_over_threshold_cnt, avg, indexes = find_values(M, threshold)

    print(f"Number of elements > {threshold}: {nums_over_threshold_cnt}")
    print(f"Average value: {avg}")
    print(f"Indexes of numbers greater than threshold: {indexes}")
