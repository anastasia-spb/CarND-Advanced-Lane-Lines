import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import find_lane_pixels


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width=50, window_height=80, margin=100):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the horizontal slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def return_lane_pixels(warped, window_centroids, window_width=50, window_height=80):
    left_lane_inds_x = []
    left_lane_inds_y = []
    right_lane_inds_x = []
    right_lane_inds_y = []
    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        center_left = window_centroids[level][0]
        left_lane_img = np.zeros_like(warped)
        left_lane_img[
        int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
        max(0, int(center_left - window_width / 2)):min(int(center_left + window_width / 2), warped.shape[1])] = 1
        center_right = window_centroids[level][1]
        right_lane_img = np.zeros_like(warped)
        right_lane_img[
        int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
        max(0, int(center_right - window_width / 2)):min(int(center_right + window_width / 2), warped.shape[1])] = 1
        # Append these indices to the lists
        left_lane_inds_y.append(left_lane_img.nonzero()[0])
        left_lane_inds_x.append(left_lane_img.nonzero()[1])
        right_lane_inds_y.append(right_lane_img.nonzero()[0])
        right_lane_inds_x.append(right_lane_img.nonzero()[1])

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds_y = np.concatenate(left_lane_inds_y)
        left_lane_inds_x = np.concatenate(left_lane_inds_x)
        right_lane_inds_y = np.concatenate(right_lane_inds_y)
        right_lane_inds_x = np.concatenate(right_lane_inds_x)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    return left_lane_inds_x, left_lane_inds_y, right_lane_inds_x, right_lane_inds_y


def draw_lane_pixels(warped, window_centroids, window_width=50, window_height=80):
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((template, zero_channel, zero_channel)), np.uint8)  # make window pixels red
        warpage = (np.dstack((warped, warped, warped)) * 255).astype(
            np.uint8)  # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0)  # overlay the original road image with window results

    # If no window centers found, just display orginal road image
    else:
        template = np.array([])
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    return output, template

def find_lane_pixels_convolution(binary_warped):
    '''
    Using convolution approach find pixels for right and left lanes
    :param binary_warped: binary image
    :return: same syntax as for find_lane_pixels from "hist" approach
    '''
    window_centroids = find_window_centroids(binary_warped)
    leftx, lefty, rightx, righty = return_lane_pixels(binary_warped, window_centroids)
    return leftx, lefty, rightx, righty, binary_warped

def find_lane_pixels_convolve_test():
    arrays_names = glob.glob('../test_images/warped/test*.npy')
    images_names = glob.glob('../test_images/test*.jpg')
    mat_names = glob.glob('../test_images/warped/M_test*.npy')
    mat_inv_names = glob.glob('../test_images/warped/Minv_test*.npy')
    window_width = 50
    window_height = 80
    for fname, img_name, mat_name, mat_inv_name in zip(arrays_names, images_names, mat_names, mat_inv_names):
        img_bin_orig = np.load(fname)
        img_undist = cv2.imread(img_name)
        mat = np.load(mat_name)
        mat_inv = np.load(mat_inv_name)
        warped = img_bin_orig[:, :, 0] / 255
        window_centroids = find_window_centroids(warped)
        output, template = draw_lane_pixels(warped, window_centroids, window_width, window_height)

        # Display the final results
        plt.imshow(output)
        plt.title('window fitting results')
        plt.show()


def compare_with_hist_approach():
    arrays_names = glob.glob('../test_images/warped/test*.npy')
    images_names = glob.glob('../test_images/test*.jpg')
    mat_names = glob.glob('../test_images/warped/M_test*.npy')
    mat_inv_names = glob.glob('../test_images/warped/Minv_test*.npy')
    window_width = 50
    window_height = 80
    for fname, img_name, mat_name, mat_inv_name in zip(arrays_names, images_names, mat_names, mat_inv_names):
        img_bin_orig = np.load(fname)
        img_undist = cv2.imread(img_name)
        mat = np.load(mat_name)
        mat_inv = np.load(mat_inv_name)
        warped = img_bin_orig[:, :, 0] / 255
        window_centroids = find_window_centroids(warped)
        output, template = draw_lane_pixels(warped, window_centroids, window_width, window_height)
        leftx, lefty, rightx, righty, out_img = find_lane_pixels.find_lane_pixels(warped, visu=True)
        output_combined = cv2.addWeighted(out_img.astype(np.uint8), 1, template, 0.5,
                                          0)  # overlay the original road image with window results

        # Display the final results
        plt.imshow(output_combined)
        plt.title('window fitting results')
        plt.show()


if __name__ == '__main__':
    find_lane_pixels_convolve_test()
    # compare_with_hist_approach()
