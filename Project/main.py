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

def visualize_polynomials(img_ref, combined_binary, left_fitx, right_fitx):

    # Visualize
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax1.set_title('Original image')
    ax1.imshow(img_ref)

    ploty = np.linspace(0, combined_binary.shape[0] - 1, combined_binary.shape[0])
    ax2.plot(left_fitx, ploty, color='green')
    ax2.plot(right_fitx, ploty, color='green')

    lineThickness = 2
    x = 50
    cv2.line(combined_binary, (x, 0), (x, combined_binary.shape[0]), (0, 255, 0), lineThickness)

    ax2.set_title('Transformed image')
    ax2.imshow(combined_binary)

    plt.show()

    return

def process_frame_pipelene(distorted_img):
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

def transform_inverse(warped, distorted_img, left_fitx, right_fitx):
    img_size = (distorted_img.shape[1], distorted_img.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, global_parameters.Minv, img_size, flags=cv2.INTER_LINEAR)
    # Combine the result with the original image
    result = cv2.addWeighted(distorted_img, 1, newwarp, 0.3, 0)
    return result


def test_on_test_images(save_intermediate = False, name_dir = ''):
    images_names = glob.glob('../test_images/test*.jpg')
    for fname in images_names:
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        result = process_frame_pipelene(img)
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

