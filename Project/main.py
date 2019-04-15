import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import Lane
import camera_calibration
import get_perspective
import combine_thresholds
import find_lane_pixels

class Parameters():
    def __init__(self):
        # camera matrix
        self.cam_mtx = None
        # distortion coefficients
        self.dist_coeff = None
        # perspektive trabsformation matrix
        self.M = np.empty()
        self.Minv = np.empty()

global_parameters = Parameters

def preprocess():
    # 0. Calibrate camera
    global_parameters.cam_mtx, global_parameters.dist_coeff = camera_calibration.get_calib_parameters()
    # 1. Get perpektive transformation matrix
    global_parameters.M = get_perspective.get_perpective_matrix()
    global_parameters.Minv = np.linalg.inv(global_parameters.M)
    return

def process_frame(distorted_img):
    img_size = (distorted_img.shape[1], distorted_img.shape[0])
    # 0. Undistort
    img_undist = cv2.undistort(distorted_img, global_parameters.cam_mtx, global_parameters.dist_coeff, None, global_parameters.cam_mtx)
    # 1. Transform image
    img_warped = cv2.warpPerspective(img_undist, global_parameters.M, img_size, flags=cv2.INTER_LINEAR)
    # 2. Color spaces and gradient
    combined_binary = combine_thresholds.convert_and_threshold(img_warped)
    # 3. Find polynomials describing the lanes
    left_fitx, right_fitx, left_fit, right_fit = find_lane_pixels.fit_polynomial(combined_binary)
    # 4. Draw lines on image
    out_img = np.dstack((combined_binary, combined_binary, combined_binary))*255
    # Plots the left and right polynomials on the lane lines
    ploty = np.linspace(0, combined_binary.shape[0] - 1, combined_binary.shape[0])
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='green')
    plt.imshow(combined_binary)
    plt.show()
    return combined_binary

if __name__ == '__main__':
    preprocess()
    img_input = cv2.imread('../test_images/straight_lines1.jpg')
    result = process_frame(img_input)

