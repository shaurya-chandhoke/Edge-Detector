"""
File: gradient.py
Author: Shaurya Chandhoke
Description: Helper file which contains the function used for Sobel filtering and Gradient processing.
"""
import cv2
import numpy as np


def gradient_process(inputImage, threshold):
    """
    Function that will apply both horizontal and vertical sobel filters to obtain the magnitude and direction of
    the image gradient.

    To reduce noise while this process occurs, a gaussian blur is first applied to smooth the image prior to running
    this function.

    :param inputImage: A 2D numpy array of the image. This array must have the Gaussian blur function applied already.
    :param threshold: A value representing the lower bound threshold to filter the image after the applied sobel filter.
    :return: The image with the Gradients as well as a 2D array representation with the Gradient angles
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.transpose(sobel_x)

    '''
    Important Disclaimer:

    Using the cv2.filter2D filter function provided by OpenCV due to the time complexity issues that had occurred with
    the original naive implementation. The deprecated implementation had a time complexity of O(n^2).

    This meant that with a sigma value of 5 and a kernel of 31x31, there were ~961 operations that had to be done for
    each pixel, severely growing as the sigma, kernel, or image size increased in size and value.
    '''
    Gradient_x = cv2.filter2D(src=inputImage, ddepth=-1, kernel=sobel_x, borderType=cv2.BORDER_REPLICATE)
    Gradient_y = cv2.filter2D(src=inputImage, ddepth=-1, kernel=sobel_y, borderType=cv2.BORDER_REPLICATE)

    Gradient = np.sqrt(np.square(Gradient_x) + np.square(Gradient_y))

    '''
    After collecting the Gradient vectors in their x and y components, obtain the Gradient magnitude and direction.
    
    Note:
    The np.arctan2 will sometimes rewrite angles with clockwise values instead of counterclockwise (-45 instead of 315).
    The modulus operation will ensure the angles stay within [0, 360] inclusively.
    '''
    Gradient_Direction = np.rad2deg(np.arctan2(Gradient_y, Gradient_x))

    Gradient_Direction = Gradient_Direction % 360

    Gradient = np.where(Gradient > threshold, Gradient, 0)

    return Gradient, Gradient_Direction
