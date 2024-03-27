import numpy as np
from PIL import Image
import time
import multiprocessing as mp


def load_image_to_array(image_file_path: str) -> np.array:
    img = Image.open(image_file_path)
    return np.asarray(img)


def save_array_as_image(image_as_array: np.array, filename: str) -> None:
    img = Image.fromarray(image_as_array)
    img.save(filename)


def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.array:
    x = np.linspace(-3.0 * sigma, 3.0 * sigma, kernel_size)
    kernel_1d = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * np.power(x, 2) / (sigma ** 2))
    return kernel_1d / np.sum(kernel_1d)


def convolve(input_image: np.array, output_image: np.array, kernel: np.array, layer: int) -> None:
    for row in range(input_image.shape[0]):
        output_image[row, :, layer] = np.convolve(input_image[row, :, layer], kernel, mode='same')
    for col in range(input_image.shape[1]):
        output_image[:, col, layer] = np.convolve(output_image[:, col, layer], kernel, mode='same')


def convolve_image(input_image: np.array, kernel: np.array) -> np.array:
    pool = mp.Pool(processes=3)
    # create copy
    convolved_image = np.zeros_like(input_image)
    for colour_index in range(convolved_image.shape[2]):
        pool.apply_async(convolve, args=(input_image, convolved_image, kernel, colour_index))
    pool.close()
    pool.join()
    return convolved_image


def run_demo():
    input_image_path = "dog.jpg"
    output_image_name = "blurry dog"
    output_image_extension = ".jpg"

    kernel_size = 5
    sigma = 3.0
    kernel = create_gaussian_kernel(kernel_size, sigma)

    image_as_array = load_image_to_array(input_image_path)

    start = time.time_ns()
    convolved_image = convolve_image(image_as_array, kernel)
    end = time.time_ns()

    print(f"convolution took: {(end - start) / 1000_000_000:.3f}s")
    save_array_as_image(convolved_image, f"alt-{output_image_name}-{kernel_size}k{sigma:.1f}s{output_image_extension}")


if __name__ == "__main__":
    run_demo()
