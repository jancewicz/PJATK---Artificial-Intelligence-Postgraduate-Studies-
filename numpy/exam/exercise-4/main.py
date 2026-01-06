import numpy as np
from numpy.typing import NDArray


def normalisation_minmax(data: NDArray):
    """
    Applies min max normalisation on numpy array.
    :param data: 2D numpy array
    :return: Modified new array with normalisation applied to each column.
    """

    # Declare function to perform minimax normalisation
    def apply_minmax(x, x_min: float, x_max: float):
        norm_x = (x - x_min) / (x_max - x_min)
        return norm_x

    # Copy data to avoid mutating original array
    data_copy = data.copy()

    minimal_vals = data_copy.min(axis=0)
    maximum_vals = data_copy.max(axis=0)

    return apply_minmax(data_copy, minimal_vals, maximum_vals)


def standardisation_zscore(data: NDArray):
    """
    Applies z score standardisation on numpy array.

    :param data: 2D numpy array
    :return: Modified new array with standardisation applied to each column.
    """

    # Declare function to perform zscore standardisation
    def apply_zscore(x, x_avg, x_sd):
        return (x - x_avg) / x_sd

    # Copy data to avoid mutating original array
    data_copy = data.copy()

    avgs = data_copy.mean(axis=0)
    sds = data_copy.std(axis=0)

    return apply_zscore(data_copy, avgs, sds)


def collect_statistics(data: NDArray) -> dict[str, dict[str, float]]:
    """
    Gathers averages, minimum values, maximum values and standard deviation for each column to array.
    Then for each unit like mass, temperature and pressure the dictionary is created with according data.
    :param data: 2D numpy array
    :return:
    """
    avgs = np.round(np.average(data, axis=0), 2)
    mins = np.round(np.min(data, axis=0), 2)
    maxes = np.round(np.max(data, axis=0), 2)
    stds = np.round(np.std(data, axis=0), 2)

    features = ["mass", "temperature", "pressure"]
    result = {
        name: {"average": avg, "min": min_val, "max": max_val, "sd": sd}
        for name, avg, min_val, max_val, sd in zip(features, avgs, mins, maxes, stds)
    }

    return result


def display_statistics(statistics: dict) -> None:
    for feature_name, feature_data in statistics.items():
        print("=====")
        print(f"{feature_name}")
        for key, val in feature_data.items():
            print(f"{key}: {val}")


if __name__ == "__main__":
    # Data: 100 measures x 3 features
    # Feature 1: mass [kg] - range 50-100
    # Feature 2: temperature [°C] - range 15-30
    # Feature 3: pressure [hPa] - range 980-1030
    data = np.random.rand(100, 3)
    data[:, 0] = data[:, 0] * 50 + 50  # mass
    data[:, 1] = data[:, 1] * 15 + 15  # temperature
    data[:, 2] = data[:, 2] * 50 + 980  # pressure

    normalised = normalisation_minmax(data)
    standardised = standardisation_zscore(data)

    # Collect statistics
    collected_statistics = [
        collect_statistics(data),
        collect_statistics(normalised),
        collect_statistics(standardised),
    ]

    for stats in collected_statistics:
        display_statistics(stats)
