"""
File: non_max_suppression.py
Author: Shaurya Chandhoke
Description: Helper file which contains the functions used for Non-Maximum Suppression
"""
import numpy as np


def imagePolish(image, threshold):
    """
    Final polish up of the image after Non-Maximum Suppression has been applied. This will convert the image to an
    unsigned 8 bit integer which will allow for display and saving.

    :param image: The pre-processed image as a 2D numpy array.
    :param threshold: The lower bound value to re-apply to the image. This is due to changing from float back to uint8.
    :return: The finalized image
    """
    scale_min = np.min(image)
    scale_max = np.max(image)

    image = np.uint8((image - scale_min) / (scale_max - scale_min) * 255)
    image = np.where(image > threshold, 255, 0)
    image = np.uint8(image)

    return image


def roundNearestDegree(deg):
    """
    A function that will round a degree value to one of 8 values respective to their sub quadrants in the unit circle.

    Because this function will be called multiple times in quick succession for each element in the Gradient Angle
    matrix, this function will be first vectorized prior to calling to allow for greater performance.

    To assign the degree value to its respective rounded quadrant angle, the following comparisons will be used:
        In total, the degree passed will round to 1 of 8 possible values:
            * 0 (also synonymous with 360)
            * 45
            * 90
            * 135
            * 180
            * 225
            * 270
            * 315

    :param deg: Element wise degree value passed from the Gradient Angle matrix
    :return: A rounded degree value based on the 8 sub quadrants of the unit circle
    """
    if (0 <= deg <= 22) or (338 <= deg <= 360):
        return 0
    elif 23 <= deg <= 67:
        return 45
    elif 68 <= deg <= 112:
        return 90
    elif 113 <= deg <= 157:
        return 135
    elif 158 <= deg <= 202:
        return 180
    elif 203 <= deg <= 247:
        return 225
    elif 248 <= deg <= 292:
        return 270
    elif 293 <= deg <= 337:
        return 315


def nonMaxSuppression(gradientMagnitudes, gradientAngles, threshold):
    """
    A helper function to perform Non-Maximum suppression. This function will first round the Gradient Angles matrix,
    then determine which pixels to compare based on the rounded values. It will perform the following comparisons:

        * If the angle is vertical (90 or 270), it means the edge is horizontal and will check pixels above and below
        * If the angle is horizontal (0 or 180), it means the edge is vertical and will check pixels left and right
        * If the angle is a Quadrant I or III angle (45 or 225), it means the edge is diagonal and will check pixels
          in those Quadrants
        * If the angle is a Quadrant II or IV angle (135 or 315), it means the edge is diagonal and will check pixels
          in those Quadrants

    :param gradientMagnitudes: A 2D numpy array of the image after the Sobel filter is applied
    :param gradientAngles: A 2D numpy array containing the direction of the Gradient after the Sobel filter is applied.
    :param threshold: An 8 bit [0-255] threshold value to clean the image after performing Non-Maximum Suppression.
    :return: The finalized image as a 2D numpy array
    """
    scaledAngles = np.int16(gradientAngles)
    bufferImage = np.zeros(shape=gradientMagnitudes.shape, dtype=gradientMagnitudes.dtype)

    # Vectorizing the function increases its efficiency
    degreeRounder = np.vectorize(roundNearestDegree)
    scaledAngles = degreeRounder(scaledAngles)

    # Creating a 1 pixel wide buffer zone across the input image to allow for diagonal pixel checking
    for i in range(1, len(scaledAngles) - 1):
        for j in range(1, len(scaledAngles[i]) - 1):
            magnitude = gradientMagnitudes[i, j]
            theta = scaledAngles[i, j]

            # Horizontal edge -- check above and below
            if theta == 90 or theta == 270:
                above = gradientMagnitudes[i - 1, j]
                below = gradientMagnitudes[i + 1, j]
                if magnitude > above and magnitude > below:
                    bufferImage[i, j] = magnitude
                else:
                    bufferImage[i, j] = 0

            # Vertical edge -- check left and right
            elif theta == 0 or theta == 180:
                left = gradientMagnitudes[i, j - 1]
                right = gradientMagnitudes[i, j + 1]

                if magnitude > left and magnitude > right:
                    bufferImage[i, j] = magnitude
                else:
                    bufferImage[i, j] = 0

            # Quadrant I and Quadrant III edge -- check both diagonal orientations
            elif theta == 45 or theta == 225:
                upperDiagonal = gradientMagnitudes[i - 1, j - 1]
                lowerDiagonal = gradientMagnitudes[i + 1, j + 1]

                if magnitude > upperDiagonal and magnitude > lowerDiagonal:
                    bufferImage[i, j] = magnitude
                else:
                    bufferImage[i, j] = 0

            # Quadrant II and Quadrant IV edge -- check both diagonal orientations
            elif theta == 135 or theta == 315:
                upperDiagonal = gradientMagnitudes[i - 1, j + 1]
                lowerDiagonal = gradientMagnitudes[i + 1, j - 1]

                if magnitude > upperDiagonal and magnitude > lowerDiagonal:
                    bufferImage[i, j] = magnitude
                else:
                    bufferImage[i, j] = 0

    finalizedImage = imagePolish(bufferImage, threshold)
    return finalizedImage
