import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import find_lane_pixels_convolution


def find_good_indexis_in_window(out_img, window_idx, nonzerox, nonzeroy, window_height, current_pos, margin=100,
                                visu=False):
    '''
    Calculate good (==nonzero) indices inside sliding window defined by input parameters
    :param out_img: binary image the same shape as input wrapped image on which result will be visualized if visu parameter set to True
    :param window_idx: current idx of window for y position calculation
    :param nonzerox: x indices of the elements that are non-zero
    :param nonzeroy: y indices of the elements that are non-zero
    :param window_height: height of the sliding window
    :param current_pos: current x position
    :param margin: width of the windows +/- margin
    :param visu: draw sliding window onto out_img
    :return: nonzero  indices inside sliding window
    '''
    # Identify window boundaries in x and y
    height = out_img.shape[0]
    win_y_low = height - (window_idx + 1) * window_height
    win_y_high = height - window_idx * window_height
    win_x_low = current_pos - margin
    win_x_high = current_pos + margin

    if visu == True:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

    # Identify the nonzero pixels in x and y within the window #
    good_inds = \
    ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

    return good_inds, out_img


def find_lane_pixels(binary_warped, visu=False):
    # Take a histogram of the bottom half of the image
    binary_filtered, histogram = get_hist(binary_warped)
    midpoint = np.int(histogram.shape[0] // 2)
    # Find the peak of the left and right halves of the histogram
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # number of sliding windows
    nwindows = 9
    # width of the windows +/- margin
    margin = 100
    # minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_filtered.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_filtered.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    if visu == True:
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped * 255, binary_warped * 255, binary_warped * 255))
    else:
        out_img = binary_warped

    # Step through the windows one by one
    for window_idx in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        good_left_inds, out_img = find_good_indexis_in_window(out_img, window_idx, nonzerox, nonzeroy, window_height,
                                                              leftx_current, margin, visu)
        good_right_inds, out_img = find_good_indexis_in_window(out_img, window_idx, nonzerox, nonzeroy, window_height,
                                                               rightx_current, margin, visu)

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def calculate_polynomial_coefficients(binary_warped, lefty, leftx, righty, rightx, ym_per_pix = 1, xm_per_pix = 1):
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty
    return left_fitx, right_fitx, left_fit, right_fit, left_fit_cr, right_fit_cr

def fit_polynomial(binary_warped, visu=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped, visu)
    # uncomment the function below and comment the find_lane_pixels function to see result of convolution approach instead
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels_convolution.find_lane_pixels_convolution(binary_warped)

    left_fitx, right_fitx, left_fit, right_fit, left_fit_cr, right_fit_cr = calculate_polynomial_coefficients(binary_warped, lefty, leftx, righty, rightx)

    ## Visualization ##
    # Colors in the left and right lane regions
    if visu == True:
        # visu wrapped image with sliding windows (result from find_lane_pixels)
        plt.imshow(out_img.astype(np.uint8))
        plt.show()

        out_img = np.dstack((binary_warped * 255, binary_warped * 255, binary_warped * 255))

        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        return left_fitx, right_fitx, left_fit, right_fit, out_img

    return left_fitx, right_fitx, left_fit, right_fit, binary_warped


def visu_histogram(binary_warped):
    binary_filtered, histogram = get_hist(binary_warped)

    # Visualize
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(20, 10))
    ax1.set_title('Binary warped image')
    ax1.imshow(binary_filtered)

    ax2.set_title('Histogram')
    ax2.plot(histogram)

    plt.show()

    return


def get_hist(binary_warped):
    [height, width] = binary_warped.shape
    histogram = np.sum(binary_warped[height // 2:height, :], axis=0)
    return binary_warped, histogram


def visualize_polynomials(img_ref, combined_binary, left_fitx, right_fitx):
    # Visualize
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax1.set_title('Original image')
    ax1.imshow(img_ref)

    ploty = np.linspace(0, combined_binary.shape[0] - 1, combined_binary.shape[0])
    ax2.plot(left_fitx, ploty, color='green')
    ax2.plot(right_fitx, ploty, color='green')

    ax2.set_title('Transformed image')
    ax2.imshow(combined_binary.astype(np.uint8))

    plt.show()

    return


def transform_inverse(warped, distorted_img, left_fitx, right_fitx, mat_inv):
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
    newwarp = cv2.warpPerspective(color_warp, mat_inv, img_size, flags=cv2.INTER_LINEAR)
    # Combine the result with the original image
    result = cv2.addWeighted(distorted_img, 1, newwarp, 0.3, 0)
    return result


def find_lane_pixels_test():
    arrays_names = glob.glob('../test_images/warped/test*.npy')
    images_names = glob.glob('../test_images/test*.jpg')
    mat_names = glob.glob('../test_images/warped/M_test*.npy')
    mat_inv_names = glob.glob('../test_images/warped/Minv_test*.npy')
    for fname, img_name, mat_name, mat_inv_name in zip(arrays_names, images_names, mat_names, mat_inv_names):
        img_bin_orig = np.load(fname)
        img_undist = cv2.imread(img_name)
        mat = np.load(mat_name)
        mat_inv = np.load(mat_inv_name)
        img_bin = img_bin_orig[:, :, 0] / 255
        left_fitx, right_fitx, left_fit, right_fit, binary_lanes = fit_polynomial(img_bin, visu=True)
        # 4. Transform back
        result = transform_inverse(img_bin, img_undist, left_fitx, right_fitx, mat_inv)
        # Plots the left and right polynomials on the lane lines
        visualize_polynomials(result, binary_lanes, left_fitx, right_fitx)


def test_histogram():
    '''
    Calculate and visualize histograms calculated on images from the previous step
    :return:
    '''
    arrays_names = glob.glob('../test_images/warped/test*.npy')
    for fname in arrays_names:
        img = np.load(fname)
        img_bin = img[:, :, 0] / 255
        visu_histogram(img_bin)


if __name__ == '__main__':
    # uncomment function below to see histograms
    # test_histogram()
    find_lane_pixels_test()
