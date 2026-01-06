def rail_fence_encrypt(text: str, num_rails: int) -> str:
    """
    Function applies rail fence encryption for given text.
    Having n number of rails, each letter of text is put into according rail in zigzag manner.
    After entering last rail, the direction of char distribution is reversed, until very first rail enters again.
    Then cycles repeats.
    :param text: Message provided by the user.
    :param num_rails: Number of rails to distribute chars in encryption method.
    :return: Encrypted message
    """
    if num_rails < 1:
        raise Exception("Liczba szyn musi być większa od 1")

    # Delete all spaces in text
    text_no_spaces = text.replace(" ", "")

    # Define rails 2D list
    rails = [[] for lst in range(1, num_rails + 1)]
    forward_direction = True
    current_rail = 0

    for char in text_no_spaces:
        if forward_direction:
            # Start adding char to according rail
            rails[current_rail].append(char)
            current_rail += 1

            # If current rail value equals to total number of rails, change direction
            if current_rail == num_rails - 1:
                forward_direction = False
                continue

        if not forward_direction:
            # Add char to rail
            rails[current_rail].append(char)
            # Because we are going backwards, decrement current_rail counter
            current_rail -= 1

            # If current rail is first one, change direction to move forward
            if current_rail == 0:
                forward_direction = True
                continue

    # Join chars in each rail list, then join n strings together for encrypted message
    encrypted_msg = "".join(["".join(rail) for rail in rails])

    return encrypted_msg


# Separate function to display necessary data for exercise requirements
def display_encryption_data(text: str, num_rails: int) -> None:
    print(f"Oryginalna wiadomość: {text}")
    print(f"Liczba szyn: {num_rails}")
    print(f"Zaszyfrowana wiadomość: {rail_fence_encrypt(text, num_rails)}")


if __name__ == "__main__":
    txt = "Podstawy pythona"
    n_rails = 3
    display_encryption_data(txt, n_rails)
