import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import Lane
import camera_calibration
import get_perspective
import combine_thresholds
import find_lane_pixels
import os
import ntpath

class Parameters():
    def __init__(self):
        # camera matrix
        self.cam_mtx = None
        # distortion coefficients
        self.dist_coeff = None

global_parameters = Parameters

def preprocess():
    # 0. Calibrate camera
    global_parameters.cam_mtx, global_parameters.dist_coeff = camera_calibration.get_calib_parameters()
    return


def process_first_frame(distorted_img):
    img_size = (distorted_img.shape[1], distorted_img.shape[0])
    # 0. Undistort
    img_undist = cv2.undistort(distorted_img, global_parameters.cam_mtx, global_parameters.dist_coeff, None, global_parameters.cam_mtx)
    # 1. Transform image
    img_warped = cv2.warpPerspective(img_undist, global_parameters.M, img_size, flags=cv2.INTER_LINEAR)
    # 2. Color spaces and gradient
    combined_binary = combine_thresholds.convert_and_threshold(img_warped)
    # 3. Find polynomials describing the lanes
    left_fitx, right_fitx, left_fit, right_fit = find_lane_pixels.fit_polynomial(combined_binary)
    # 4. Transform back
    result = transform_inverse(combined_binary, distorted_img, left_fitx, right_fitx)
    # Plots the left and right polynomials on the lane lines
    visualize_polynomials(result, combined_binary, left_fitx, right_fitx)
    return combined_binary


def test_on_test_images(save_intermediate = False, name_dir = ''):
    images_names = glob.glob('../test_images/test*.jpg')
    for fname in images_names:
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        result = process_first_frame(img)
        if save_intermediate == True:
            save_intermediate_result(result, name_dir, ntpath.basename(fname))

def save_intermediate_result(img, name_dir, img_name):
    directory = '../test_images/' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + img_name
    cv2.imwrite(directory, img.astype('uint8') * 255)


def main():
    preprocess()
    img_input = cv2.imread('../test_images/straight_lines1.jpg')
    result = process_frame_pipelene(img_input)

if __name__ == '__main__':
    main()
    test_on_test_images(save_intermediate = True, name_dir = 'warped')

