import numpy as np
from PIL import Image
import time
import multiprocessing as mp
import pyopencl as cl


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


def convolve_row(input_image: np.array, output_image: np.array, kernel: np.array, row: int, layer: int) -> None:
    output_image[row, :, layer] = np.convolve(input_image[row, :, layer], kernel, mode='same')


def convolve_col(input_image: np.array, output_image: np.array, kernel: np.array, col: int, layer: int) -> None:
    output_image[:, col, layer] = np.convolve(input_image[:, col, layer], kernel, mode='same')


def convolve_image(input_image: np.array, kernel: np.array) -> np.array:
    pool = mp.Pool(processes=16)
    convolved_image = np.zeros_like(input_image)
    for colour_index in range(convolved_image.shape[2]):

        for row in range(input_image.shape[0]):
            pool.apply_async(convolve_row, args=(input_image, convolved_image, kernel, row, colour_index))

        for col in range(input_image.shape[1]):
            pool.apply_async(convolve_col, args=(input_image, convolved_image, kernel, col, colour_index))
    pool.close()
    pool.join()

    return convolved_image


def run_demo():
    # input_image_path = "dog.jpg"
    # output_image_name = "blurry dog"
    # output_image_extension = ".jpg"
    #
    # kernel_size = 5
    # sigma = 3.0
    # kernel = create_gaussian_kernel(kernel_size, sigma)
    #
    # image_as_array = load_image_to_array(input_image_path)
    #
    # start = time.time_ns()
    # convolved_image = convolve_image(image_as_array, kernel)
    # end = time.time_ns()
    #
    # print(f"convolution took: {(end - start) / 1000_000_000:.3f}s")
    # save_array_as_image(convolved_image, f"alt-{output_image_name}-{kernel_size}k{sigma:.1f}s{output_image_extension}")
    rng = np.random.default_rng()
    a_np = rng.random(50000, dtype=np.float32)
    b_np = rng.random(50000, dtype=np.float32)

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

    prg = cl.Program(ctx, """
    __kernel void sum(
        __global const float *a_g, __global const float *b_g, __global float *res_g)
    {
      int gid = get_global_id(0);
      res_g[gid] = a_g[gid] + b_g[gid];
    }
    """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
    knl = prg.sum  # Use this Kernel object for repeated calls
    knl(queue, a_np.shape, None, a_g, b_g, res_g)

    res_np = np.empty_like(a_np)
    cl.enqueue_copy(queue, res_np, res_g)

    # Check on CPU with Numpy:
    error_np = res_np - (a_np + b_np)
    print(f"Error:\n{error_np}")
    print(f"Norm: {np.linalg.norm(error_np):.16e}")
    assert np.allclose(res_np, a_np + b_np)


if __name__ == "__main__":
    run_demo()
