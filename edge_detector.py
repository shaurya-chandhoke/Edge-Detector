"""
File: edge_detector.py
Author : Shaurya Chandhoke
Description: Command line script that takes an image path as input and processes the image as output
"""
import argparse
import cv2
import numpy as np
import time

from src.gaussian import gaussian_process
from src.gradient import gradient_process
from src.non_max_suppression import nonMaxSuppression


def finish(originalImg, grayScaleImage, gaussianImg, gradientImg, nonmaxImage, timeElapsed, nosave, noshow):
    """
    The final stages of the program. This function will display the images and/or write them to files as well as
    provide an execution time.

    :param originalImg: The original image when running the program
    :param grayScaleImage: The image after it has been read and converted into grayscale for the program
    :param gaussianImg: The Grayscale image after a Gaussian blur of a specified sigma value has been applied
    :param gradientImg: The Gaussian image after the Sobel filters and threshold value have been applied.
    :param nonmaxImage: The Gradient image after Non-Maximum Suppression has been applied.
    :param timeElapsed: The total execution time observed
    :param nosave: Flag that determines whether to save the images to their respective files
    :param noshow: Flag that determines whether to display the images as output
    """

    if (noshow is True) and (nosave is True):
        print("(BOTH FLAGS ON) Recommend disabling either --nosave or --quiet to capture processed images")
        return 0

    print("=" * 40)
    print("Rendering Images...")

    warning = '''
    A potential divide by 0 issue was noticed while rescaling the images. 
    This may be due to a high sigma value and you may get an entirely black image. 
    It's suggested you choose a lower sigma value and try again.
    '''

    # Scaling remaining images to unsigned 8 bit integers to allow displaying and writing in grayscale
    scale_min = np.min(gaussianImg)
    scale_max = np.max(gaussianImg)
    gaussianImg = np.uint8((gaussianImg - scale_min) / (scale_max - scale_min) * 255)

    scale_min = np.min(gradientImg)
    scale_max = np.max(gradientImg)
    gradientImg = np.uint8((gradientImg - scale_min) / (scale_max - scale_min) * 255)

    # Print warning message in case divide by 0 is detected
    if scale_min == 0 and scale_max == 0:
        print(warning)

    if noshow is False:
        print("(DISPLAY ON) The ESC key will close all pop ups")
        image1 = cv2.imshow("Original Image", originalImg)
        image2 = cv2.imshow("Grayscale", grayScaleImage)
        image3 = cv2.imshow("Gaussian Filtered", gaussianImg)
        image4 = cv2.imshow("Sobel Filtered", gradientImg)
        image5 = cv2.imshow("Non-Maximum Suppression", nonmaxImage)
        cv2.waitKey(0)

    if nosave is False:
        print("(IMAGE SAVE ON) Images are being written to the ./out/ folder")
        cv2.imwrite("./out/step_0_grayscale_result.jpg", grayScaleImage)
        cv2.imwrite("./out/step_1_gaussianfiltered_result.jpg", gaussianImg)
        cv2.imwrite("./out/step_2_sobelfiltered_result.jpg", gradientImg)
        cv2.imwrite("./out/step_3_nonmaxsuppression_result.jpg", nonmaxImage)

    print("(DONE): You may want to rerun the program with the --help flag for more options to fine tune the program")
    print("=" * 40)
    print("Time to Process Image: {} seconds.".format(timeElapsed))


def start(image, sigma, threshold):
    """
    Starter function responsible for beginning the process for obtaining edges

    :param image: The input image as a 2D numpy array
    :param sigma: The sigma value used for generating the Gaussian filter
    :param threshold: The minimum threshold used to filter out noisy pixels during Sobel filtering
    :return: Processed images to be displayed or written
    """

    print("Please wait, processing image and returning output...\n")

    print("(Step 1) Start: Applying Gaussian filter to input image")
    gaussianImage, shape = gaussian_process(image, sigma)
    print("(Step 1) Complete: Applying Gaussian filter to input image [sigma={}, kernel shape={}]".format(sigma, shape))

    print("\n(Step 2) Start: Applying Sobel filter to resulting image")
    gradientImage, gradientAngles = gradient_process(gaussianImage, threshold)
    print("(Step 2a) Complete: Applying Sobel filter to resulting image. [sobel threshold: {}]".format(threshold))
    print("(Step 2b) Complete: Obtained Gradient magnitudes and directions")

    print("\n(Step 3) Start: Applying Non-Maximum Suppression to resulting image")
    nonMaxSuppressedImage = nonMaxSuppression(gradientImage, gradientAngles, threshold)
    print("(Step 3) Complete: Applying Non-Maximum Suppression to resulting image\n")

    return gaussianImage, gradientImage, nonMaxSuppressedImage


def main():
    """
    Beginning entry point into the edge detection program.
    It will first perform prerequisite steps prior to starting the intended program.
    Upon parsing the command line arguments, it will trigger the start function
    """
    parser = argparse.ArgumentParser(prog="edge_detector.py",
                                     description="Given the path to an image, this program will process the image " +
                                                 "with edges detected.", usage="%(prog)s [imgpath] [flags]")

    parser.add_argument("imgpath", help="The file path of the image.", type=str)

    parser.add_argument("-s", "--sigma", help="The sigma value for the Gaussian function. Default value is 1",
                        type=int, default=1)

    parser.add_argument("-t", "--threshold", help="The minimum value for sobel filter thresholding. Range is [0-255]." +
                                                  " Default is 40.", type=int, default=40)

    parser.add_argument("-n", "--nosave", help="If passed, the images will not be written to a file. By default, " +
                                               "images are written.", action="store_true")

    parser.add_argument("-q", "--quiet", help="If passed, the images will not be displayed. By default, the images " +
                                              "will be displayed.", action="store_true")

    args = parser.parse_args()
    imgpath = args.imgpath
    sigma = args.sigma
    threshold = args.threshold
    nosave = args.nosave
    noshow = args.quiet

    originalImage = cv2.imread(imgpath)
    inputImage = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    ADVICE = "rerun with the (-h, --help) for more information."

    if (inputImage is None) or (originalImage is None):
        print("Error: Cannot open image.\nPlease check if the path is written correctly and try again or " + ADVICE)
        return -1

    if sigma <= 0:
        print("Error: Sigma value cannot be less than 1.\nPlease try again or " + ADVICE)
        return -1

    if 0 > threshold or threshold > 255:
        print("Error: Threshold range is [0-255].\nPlease ensure the threshold is within the range or " + ADVICE)
        return -1

    np.seterr(all="ignore")
    START_TIME = time.time()

    gaussianImage, gradientImage, nonMaxSuppressedImage = start(inputImage, sigma, threshold)

    ELAPSED_TIME = time.time() - START_TIME

    finish(originalImage, inputImage, gaussianImage, gradientImage, nonMaxSuppressedImage, ELAPSED_TIME, nosave, noshow)


if __name__ == "__main__":
    main()
