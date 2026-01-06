def check_age(age: int) -> None:
    """
    Function check users age based on provided input. If age is above 18,
    functions prints that user is grown-up, else function prints that user is under-aged.
    :param age: Provided age as integer.
    :return: None
    """
    if age >= 18:
        print("Jesteś pełnoletni/pełnoletnia.")
    else:
        print("Nie jesteś jeszcze pełnoletni/pełnoletnia.")


if __name__ == "__main__":
    age = int(input("Podaj swój wiek (wartość liczbowa w latach): "))
    check_age(age)
