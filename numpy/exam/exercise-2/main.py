import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.typing import NDArray


def convert_img_to_np_arr(img_path: str) -> NDArray:
    """
    Converts provided image - specified as a path to a .img file to numpy array.
    :param img_path: path to image as str.
    :return: converted image to NDArray.
    """
    image = Image.open(img_path)
    return np.array(image)


if __name__ == "__main__":
    broken_img_path = "santa_zepsuty.jpg"
    broken_img: NDArray = convert_img_to_np_arr(broken_img_path)

    print("Avg RGB values analysis")
    print(f"(R): {broken_img[:, :, 0].mean():.2f}")
    print(f"(G): {broken_img[:, :, 1].mean():.2f}")
    print(f"(B): {broken_img[:, :, 2].mean():.2f}")

    r_avg_value = broken_img[:, :, 0].mean()
    # Green value is too high, and whole picture is too green-ish
    g_avg_value = broken_img[:, :, 1].mean()
    b_avg_value = broken_img[:, :, 2].mean()

    fixed_img = broken_img.copy().astype(np.float32)

    # Compute global average of correct R ang B and set it to target
    target_green_avg = np.average([r_avg_value, b_avg_value])
    target_green_scaled = target_green_avg / g_avg_value

    fixed_img[:, :, 1] *= target_green_scaled
    fixed_img = np.clip(fixed_img, 0, 255).astype(np.uint8)

    # Display images
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(broken_img)
    axes[0].set_title("Zepsuty obraz")
    axes[0].axis("off")
    axes[1].imshow(fixed_img)
    axes[1].set_title("Naprawiony obraz")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
