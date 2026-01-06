import numpy as np
from numpy.typing import NDArray


def count_avg_temp_per_location(temperatures: NDArray) -> list[float]:
    """
    Function count average temperature in each location in 2D numpy array.
    :param temperatures: 2D numpy array where each row is next day, and each column represents different location.
    :return: list of numpy floats representing average value for each location.
    """
    return np.round(temperatures.mean(axis=0), 2).tolist()


def find_highest_avg_temp_day(temperatures: NDArray) -> tuple[int, float]:
    """
    Calculates what's the highest average day temperature within 2D numpy array of temperatures.
    :param temperatures: 2D numpy array where each row is next day, and each column represents different location.
    :return: A tuple with index of the row from 2D array with the highest average temperature and highest temperature
    value as float.
    """
    averages = temperatures.mean(axis=1)
    highest_day_idx = int(averages.argmax())

    highest_day_avg = float(np.round(averages[highest_day_idx], 2))

    return highest_day_idx, highest_day_avg


def mark_and_change_outlier_values_with_median(
    temperatures: NDArray,
) -> tuple[NDArray, int]:
    """
    Function computes median value for each locality. Then masks values that are below lower threshold and above
    upper threshold. Those outlier values are being changed with according locality median temperature value.
    :param temperatures: 2D numpy array where each row is next day, and each column represents different location.
    :return: A tuple with modified temperature 2D array with outliers values replaced by locality median temperature and
    total count of all outlier values that occurred in 2D.
    """

    # Copy array so we avoid original array mutations issues
    temperatures_arr_copy = temperatures.copy()

    lower_threshold = 10
    upper_threshold = 35

    # Calculate median per each column (locality)
    localities_medians = np.median(temperatures_arr_copy, axis=0)

    # Create boolean mask for outlier values below and above threshold
    outlier_vals_mask = (temperatures_arr_copy < lower_threshold) | (
        temperatures_arr_copy > upper_threshold
    )

    # Count every 'truthy' outlier value represented as 1 in numpy
    outlier_values_cnt = int(np.count_nonzero(outlier_vals_mask))
    temperatures_arr_copy[outlier_vals_mask] = localities_medians[
        np.where(outlier_vals_mask)[1]
    ]

    # Return modified array with outlier values changed to according localities median temperature value, and total
    # count of outlier values
    return temperatures_arr_copy, outlier_values_cnt


def calc_global_avg(temperatures: NDArray) -> float:
    """
    Calculates global average by flattening 2D numpy array.
    :param temperatures: 2D numpy array where each row is next day, and each column represents different location.
    :return: global average as float
    """
    temperatures_flatten = temperatures.flatten()

    return float(np.round(np.average(temperatures_flatten), 2))


def calc_global_sd(temperatures: NDArray) -> float:
    """
    Function calculate global standard deviation for temperatures array.
    :param temperatures: 2D numpy array where each row is next day, and each column represents different location.
    :return: global standard deviation value as float.
    """
    temperatures_flatten = temperatures.flatten()

    return float(np.round(np.std(temperatures_flatten), 2))


def main() -> None:
    np.random.seed(42)
    # Generating data (30 days x 4 locations)
    temperatures = np.random.normal(loc=22, scale=5, size=(30, 4))
    temperatures = np.round(temperatures, 1)

    average_temp_per_location = count_avg_temp_per_location(temperatures)

    highest_avg_temp_day_idx, highest_avg_temp = find_highest_avg_temp_day(temperatures)

    temperatures_changed, num_outlier_values = (
        mark_and_change_outlier_values_with_median(temperatures)
    )

    print("=== STATYSTYKI PRZED KOREKTĄ ===")
    print(f"Średnie temperatury w lokalizacjach: {average_temp_per_location}")
    print(
        f"Dzień z najwyższą średnią temperaturą: {highest_avg_temp_day_idx + 1} ({highest_avg_temp}°C)"
    )
    print(f"Liczba wartości odstających: {num_outlier_values}")

    global_temp_avg_before_corrections = calc_global_avg(temperatures)
    global_temp_avg_after_corrections = calc_global_avg(temperatures_changed)

    global_sd_before_corrections = calc_global_sd(temperatures)
    global_sd_after_corrections = calc_global_sd(temperatures_changed)

    print("=== STATYSTYKI PO KOREKCIE ===")
    print(
        f"Średnia globalna: {global_temp_avg_before_corrections}°C -> {global_temp_avg_after_corrections}°C"
    )
    print(
        f"Odchylenie standardowe: {global_sd_before_corrections}°C -> {global_sd_after_corrections}°C"
    )


if __name__ == "__main__":
    main()
