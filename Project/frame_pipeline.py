import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import find_lane_pixels_convolution
import find_lane_pixels
import combine_thresholds
import calculate_curvation
import get_perspective


def mask_image(input_img, mask_points, inside):
    '''
    Mask all pixels which are not inside ROI
    '''
    vertices_mask = np.array([mask_points], dtype=np.int32)
    masked_img = get_perspective.region_of_interest(input_img, vertices_mask, inside)
    return masked_img

def frame_pipeline(orig_img, M, Minv, cam_mtx, dist_coeff, mask_points, mask_points_inside, method = "histogram"):
    '''
    Implements line searching pipeline
    :param orig_img:
    :param M: matrix used in perspective transformation of the first frame
    :param Minv: inverse matrix used in perspective transformation of the first frame
    :return: return the image with marked lane area, curvation and
             vehicle position info in form of text on the image frame
    '''
    # 0. Undistort image
    img_undist = cv2.undistort(orig_img, cam_mtx, dist_coeff, None, cam_mtx)
    # 1. Combine results of Sobel and magnitude thresholds for grayscale and s channel (hls)
    thresholds = combine_thresholds.Thresholds()
    combined_binary = combine_thresholds.convert_and_threshold(img_undist, thresholds, visu = False)
    # 2. Wrap combined binary
    combined_binary_masked = mask_image(combined_binary, mask_points, False) # mask outside regions
    combined_binary_masked = mask_image(combined_binary_masked, mask_points_inside, True)  # mask inner region
    img_size = (combined_binary_masked.shape[1], combined_binary_masked.shape[0])
    binary_warped = cv2.warpPerspective(combined_binary_masked, M, img_size, flags=cv2.INTER_LINEAR)
    # 3. Find lane pixels
    if(method == "convolution"):
        leftx, lefty, rightx, righty, out_img = find_lane_pixels_convolution.find_lane_pixels_convolution(binary_warped)
    else: # method == "histogram"
        leftx, lefty, rightx, righty, out_img = find_lane_pixels.find_lane_pixels(binary_warped)
    # 4. Calculate polynomial coefficients
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fitx, right_fitx, left_fit, right_fit, left_fit_cr, right_fit_cr = find_lane_pixels.calculate_polynomial_coefficients(binary_warped, lefty, leftx, righty, rightx, ym_per_pix, xm_per_pix)
    # 5. Calculate curvation coefficient
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_curverad, right_curverad = calculate_curvation.measure_curvature_pixels(ploty, left_fit_cr, right_fit_cr, ym_per_pix)
    # 6. Calculate deviation from lane center
    deviation = calculate_curvation.get_deviation_from_center(ploty, left_fit_cr, right_fit_cr, ym_per_pix, xm_per_pix)
    # 7. transform inverse left and right pixels fit
    orig_img_marked_lanes = find_lane_pixels.transform_inverse(binary_warped, orig_img, left_fitx, right_fitx, Minv)
    #8. Print text on image
    result = calculate_curvation.print_data_on_image(deviation, left_curverad, right_curverad, orig_img_marked_lanes)
    return result


def first_frame_pipeline(orig_img, cam_mtx, dist_coeff):
    '''
    For the first frame choose points for calculating perpective matrix
    and ROI points
    '''
    # 0. Undistort image
    img_undist = cv2.undistort(orig_img, cam_mtx, dist_coeff, None, cam_mtx)
    #1. Choose src and roi points
    M, Minv, img_with_marked_area = get_perspective.calculate_perpective_matrix(img_undist)
    mask_points = get_perspective.choose_points(img_undist, draw_help_lines=False)
    mask_points_inside = get_perspective.choose_points(img_undist, draw_help_lines=False)
    #2. Continue the same pipeline as for the rest of frames
    result = frame_pipeline(orig_img, M, Minv, cam_mtx, dist_coeff, mask_points, mask_points_inside)
    return M, Minv, mask_points, mask_points_inside, result




