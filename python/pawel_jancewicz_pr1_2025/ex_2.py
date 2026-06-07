def count_vowels(text: str) -> None:
    """
    Function counts total occurrence of each vowel in provided sentence, then print its results.
    :param text: Sentence provided by the user input.
    :return: None
    """
    lowercase_txt = text.lower()

    vowels_cnt: dict[str, int] = {
        "a": 0,
        "e": 0,
        "i": 0,
        "o": 0,
        "u": 0,
        "y": 0,
    }

    for char in lowercase_txt:
        if char in vowels_cnt:
            vowels_cnt[char] += 1

    for key, val in vowels_cnt.items():
        if val != 0:
            print(f"{key}: {val}")


if __name__ == "__main__":
    text = input("Podaj wybrane zdanie, aby policzyć samogłoski: ")
    count_vowels(text)
