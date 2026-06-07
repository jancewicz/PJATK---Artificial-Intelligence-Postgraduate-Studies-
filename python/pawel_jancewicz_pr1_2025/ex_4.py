def group_words_by_length(path: str) -> dict[int, list[str]]:
    """
    Function reads .txt file line by line, and collects words with the same length in dictionary.
    :param path: Path to txt file.
    :return: Dictionary that is sorted by key, with word length as key, and list of words with that length as value.
    """

    # Open file and append each line into list
    with open(path, "r") as file:
        lines = [line.rstrip() for line in file]

    # Init dict
    words_by_len: dict[int, list[str]] = {}

    # Iterate over lines list
    for line in lines:
        word_len = len(line)
        # If word length is not yet in dict, create new empty list, and add record.
        if word_len not in words_by_len.keys():
            words_by_len[word_len] = []
            words_by_len[word_len].append(line)
        # If word length is already in dict, append the same length word into the list.
        else:
            words_by_len[word_len].append(line)

    # Return sorted dictionary
    return dict(sorted(words_by_len.items()))


def write_results_to_file(words_by_length: dict[int, list[str]]) -> None:
    """
    Function reads dict of grouped words by length, then writes to file processed text.
    :param words_by_length:
    :return:
    """

    # w+ ensures that if file does not exist, it will be created
    with open("result.txt", "w+") as file:
        for key, value in words_by_length.items():
            line = f"Długość {key}: {", ".join(value)}\n"
            file.write(line)


if __name__ == "__main__":
    words_by_length = group_words_by_length("example.txt")
    write_results_to_file(words_by_length)
