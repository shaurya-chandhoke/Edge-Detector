"""
File: gradient.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for Sobel filtering and Gradient processing.
"""
import cv2
import numpy as np


def gradient_process(input_image, threshold):
    """
    Function that will apply both horizontal and vertical sobel filters to obtain the magnitude and direction of
    the image gradient.

    To reduce noise while this process occurs, a gaussian blur is first applied to smooth the image prior to running
    this function.

    :param input_image: A 2D numpy array of the image. This array must have the Gaussian blur function applied already.
    :param threshold: A value representing the lower bound threshold to filter the image after the applied sobel filter.
    :return: The image with the Gradients as well as a 2D array representation with the Gradient angles
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.transpose(sobel_x)

    """
    Important Disclaimer:

    Using the cv2.filter2D filter function provided by OpenCV due to the time complexity issues that had occurred with
    the original naive implementation. The deprecated implementation had a time complexity of O(n^2).

    This meant that with a sigma value of 5 and a kernel of 31x31, there were ~961 operations that had to be done for
    each pixel, severely growing as the sigma, kernel, or image size increased in size and value.
    """
    gradient_x = cv2.filter2D(src=input_image, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REPLICATE)
    gradient_y = cv2.filter2D(src=input_image, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REPLICATE)

    gradient = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    """
    After collecting the Gradient vectors in their x and y components, obtain the Gradient magnitude and direction.
    
    Note:
    The np.arctan2 will sometimes rewrite angles with clockwise values instead of counterclockwise (-45 instead of 315).
    The modulus operation will ensure the angles stay within [0, 360] inclusively.
    """
    gradient_direction = np.rad2deg(np.arctan2(gradient_y, gradient_x))

    gradient_direction = gradient_direction % 360

    gradient = np.where(gradient > threshold, gradient, 0)

    return gradient, gradient_direction
