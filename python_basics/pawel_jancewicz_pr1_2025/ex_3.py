import math


def is_prime(n: int) -> bool:
    """
    Checks whether provided number is prime number.
    :param n: User provided number
    :return: Boolean value based on number being prime or not.
    """

    # First error fix: in prime algorithm we check whether 1 is greater or equal to n, if yes we return False.
    if n <= 1:
        return False

    # Second error fix: Calculate square root of n and check range of numbers between 2 and square root of n.
    # This step was skipped.
    sqrt_n = int(math.sqrt(n))
    # Third error fix: sqrt_n + 1 because range excludes upper band, adding + 1 ensures that we fully loop up to sqrt_n
    for i in range(2, sqrt_n + 1):
        if n % i == 0:
            return False

    return True


if __name__ == "__main__":
    low_range_value = int(input("Podaj dolną granicę zakresu: "))
    upper_range_value = int(input("Podaj górną granicę zakresu: "))

    print(f"Liczby pierwsze w zakresie {low_range_value}-{upper_range_value}")

    # Loop can be simplified to list comprehension
    # Fix for third error is applied here too to expand upper boundary by 1
    primes = [
        str(num)
        for num in range(low_range_value, upper_range_value + 1)
        if is_prime(num)
    ]

    if len(primes) > 0:
        print(", ".join(primes))
    else:
        print("Brak liczb pierwszych w podanym zakresie")
