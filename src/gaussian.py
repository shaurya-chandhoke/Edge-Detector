"""
File: gaussian.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for Gaussian blurring.
"""
from math import pi, exp, pow
import numpy as np
import cv2


def gaussian_process(image, sigma):
    """
    Given a sigma value, this function will instantiate a 2D square numpy kernel and apply a Gaussian function to
    normalize its values.

    The dimensions of the kernel is created using the following rule:
        2 * (sigma * 3) + 1

    In other words, the dimension of one half of a side of the kernel will be:
        3 * sigma

    Example:
    A sigma value of 1 yields a square kernel with a single side of length 7 (from the center, 3 units on both sides
    which makes 6 + (the center row) = 7). Thus, the dimensions of the kernel are 7x7

    :param image: The input image represented as a 2D numpy array
    :param sigma: The sigma value for the kernel and gaussian function to be applied.
    :return: The 2D normalized array
    """
    kernel_size = (2 * (sigma * 3)) + 1
    kernel_filter = np.empty((kernel_size, kernel_size), dtype=np.float64)

    kernel_size_half = kernel_size // 2
    kernel_k_index = -(sigma * 3)
    kernel_l_index = (sigma * 3) + 1
    kernel_indices = np.arange(kernel_k_index, kernel_l_index, 1)

    gaussian_constant = (1 / (2 * pi * pow(sigma, 2)))

    for K in kernel_indices:
        for L in kernel_indices:
            gaussian_exponent = exp(-(pow(K, 2) + pow(L, 2)) / (2 * pow(sigma, 2)))

            # Add K, L to halfway kernel_size to bring indices to values compatible with python arrays [0,1,2,3...]
            kernel_filter[K + kernel_size_half, L + kernel_size_half] = gaussian_constant * gaussian_exponent

    """
    Important Disclaimer:
    
    Using the cv2.filter2D filter function provided by OpenCV due to the time complexity issues that had occurred with
    the original naive implementation. The deprecated implementation had a time complexity of O(n^2).
    
    This meant that with a sigma value of 5 and a kernel of 31x31, there were ~961 operations that had to be done for
    each pixel, severely growing as the sigma, kernel, or image size increased in size and value.
    
    ========================
    
    Note:
    The cv2.filter2D returns a response of data type np.uint8 (unsigned 8 bit integer -- values range from [0-255]).
    As a result, to continue with Gradient computation, the input image must be converted to a float data type.
    """
    final_image = cv2.filter2D(src=image, ddepth=-1, kernel=kernel_filter, borderType=cv2.BORDER_REPLICATE)
    final_image = np.float64(final_image)

    return final_image, kernel_filter.shape
