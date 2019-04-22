import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def measure_curvature_pixels(ploty, left_fit, right_fit, ym_per_pix=30 / 720):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    return left_curverad, right_curverad


def get_deviation_from_center(ploty, left_fit, right_fit, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    # Define y-value where we want calculate the deviation
    y_eval = np.max(ploty)
    # calculate left value for previously chosen y value
    left = np.polyval(left_fit, y_eval * ym_per_pix)
    right = np.polyval(right_fit, y_eval * ym_per_pix)
    # the point from which we calculate deviation
    middle = (len(ploty) / 2) * xm_per_pix
    deviation = middle - left

    return deviation


def print_data_on_image(deviation, left_curverad, right_curverad, image):
    '''
    This function modifies input image by putting text on it
    :param deviation:
    :param left_curverad:
    :param right_curverad:
    :param image:
    :return:
    '''
    text_radius = 'Radius of Curvature = {:.2f} (m)'.format(left_curverad)
    cv2.putText(image, text_radius, (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255),
                2, lineType=cv2.LINE_AA)
    text_deviation = 'Vehicle is {:.2f} m left on center'.format(deviation)
    cv2.putText(image, text_deviation, (20, 130), cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return image
