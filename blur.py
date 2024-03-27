import numpy as np
from PIL import Image
import time


def load_image_to_array(image_file_path: str) -> np.array:
    img = Image.open(image_file_path)
    return np.asarray(img)


def save_array_as_image(image_as_array: np.array, filename: str) -> None:
    img = Image.fromarray(image_as_array)
    img.save(filename)


def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.array:
    x = np.linspace(-3.0 * sigma, 3.0 * sigma, kernel_size)
    kernel_1d = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * np.power(x, 2) / (sigma ** 2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_sum = np.sum(kernel_2d)  # this is the total area of the kernel, it should be 1.0 so no power is lost
    return kernel_2d / kernel_sum  # we normalise for the power of the kernel


def convolve(a: np.array, b: np.array) -> float:
    # the matrices must be the same size
    if a.shape != b.shape:
        return 0.0

    # they must also be 2-dimensional
    if len(a.shape) != 2 and len(b.shape) != 2:
        return 0.0

    multiplied = np.multiply(a, b)

    return np.sum(multiplied)


def convolve_image(matrix: np.array, kernel: np.array) -> np.array:
    kernel_length = kernel.shape[0]
    kernel_offset = (kernel_length - 1) // 2
    # create copy
    convolved_image = np.zeros_like(matrix)
    for colour_index in range(convolved_image.shape[2]):
        for row in range(kernel_offset, matrix.shape[0] - kernel_offset):
            for col in range(kernel_offset, matrix.shape[1] - kernel_offset):
                convolved_image[row, col, colour_index] = convolve(
                    matrix[row - kernel_offset:row + kernel_offset + 1, col - kernel_offset:col + kernel_offset + 1,
                    colour_index],
                    kernel
                )

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
    save_array_as_image(convolved_image, f"{output_image_name}-{kernel_size}k{sigma:.1f}s{output_image_extension}")


if __name__ == "__main__":
    run_demo()
