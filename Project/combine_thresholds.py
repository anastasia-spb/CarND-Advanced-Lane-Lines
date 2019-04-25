import numpy as np
import cv2
import os
import glob
import ntpath
import matplotlib.pyplot as plt


class Thresholds:
    def __init__(self, grayscale_abs_bottom=20, grayscale_abs_upper=60,
                 grayscale_magnitude_bottom=70, grayscale_magnitude_upper=100,
                 channel_abs_bottom=35, channel_abs_upper=50,
                 channel_magnitude_bottom=60, channel_magnitude_upper=70):
        self.grayscale_abs_bottom = grayscale_abs_bottom
        self.grayscale_abs_upper = grayscale_abs_upper
        self.grayscale_magnitude_bottom = grayscale_magnitude_bottom
        self.grayscale_magnitude_upper = grayscale_magnitude_upper
        self.channel_abs_bottom = channel_abs_bottom
        self.channel_abs_upper = channel_abs_upper
        self.channel_magnitude_bottom = channel_magnitude_bottom
        self.channel_magnitude_upper = channel_magnitude_upper

    def set_GRAY_abs_upper(self, x):
        self.grayscale_abs_upper = x

    def set_GRAY_abs_bottom(self, x):
        self.grayscale_abs_bottom = x

    def set_GRAY_magnitude_bottom(self, x):
        self.grayscale_magnitude_bottom = x

    def set_GRAY_magnitude_upper(self, x):
        self.grayscale_magnitude_upper = x

    def set_channel_abs_bottom(self, x):
        self.channel_abs_bottom = x

    def set_channel_abs_upper(self, x):
        self.channel_abs_upper = x

    def set_channel_magnitude_upper(self, x):
        self.channel_magnitude_upper = x

    def set_channel_magnitude_bottom(self, x):
        self.channel_magnitude_bottom = x


def choose_threshold_value(input_img):
    img_window_name = 'image'
    trackbars_window_name = 'trackbars'
    cv2.namedWindow(img_window_name)
    cv2.namedWindow(trackbars_window_name, cv2.cv2.WINDOW_NORMAL)
    #cv2.resizeWindow(trackbars_window_name, 600, 600)

    current_thresholds = Thresholds()  # start with init values
    combined_binary = convert_and_threshold(input_img, current_thresholds, visu=False)

    # create trackbars for threshold change
    cv2.createTrackbar('GRAY Abs B', trackbars_window_name, current_thresholds.grayscale_abs_bottom, 150, current_thresholds.set_GRAY_abs_bottom)
    cv2.createTrackbar('GRAY Abs U', trackbars_window_name, current_thresholds.grayscale_abs_upper, 150, current_thresholds.set_GRAY_abs_upper)
    cv2.createTrackbar('GRAY Mag B', trackbars_window_name, current_thresholds.grayscale_magnitude_bottom, 150, current_thresholds.set_GRAY_magnitude_bottom)
    cv2.createTrackbar('GRAY Mag U', trackbars_window_name, current_thresholds.grayscale_magnitude_upper, 150, current_thresholds.set_GRAY_magnitude_upper)
    cv2.createTrackbar('Ch Abs B', trackbars_window_name, current_thresholds.channel_abs_bottom, 150, current_thresholds.set_channel_abs_bottom)
    cv2.createTrackbar('Ch Abs U', trackbars_window_name, current_thresholds.channel_abs_upper, 150, current_thresholds.set_channel_abs_upper)
    cv2.createTrackbar('Ch Mag B', trackbars_window_name, current_thresholds.channel_magnitude_bottom, 150, current_thresholds.set_channel_magnitude_bottom)
    cv2.createTrackbar('Ch Mag U', trackbars_window_name, current_thresholds.channel_magnitude_upper, 150, current_thresholds.set_channel_magnitude_upper)

    threshed = cv2.adaptiveThreshold(combined_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

    while (1):
        cv2.imshow(img_window_name, threshed)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        combined_binary = convert_and_threshold(input_img, current_thresholds, visu=False)
        threshed = cv2.adaptiveThreshold(combined_binary, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)

    cv2.destroyAllWindows()


def abs_sobel_thresh(image, orient='x', thresh=(0, 255)):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy
    binary_output = np.zeros_like(scaled_sobel)
    # apply the threshold
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the binary image
    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output


def get_S_channel(img_rgb):
    hls_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    s_channel_img = hls_img[:, :, 2]
    return s_channel_img


def get_V_channel_Luv(img_rgb):
    luv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
    v_channel_img = luv_img[:, :, 2]
    return v_channel_img


def get_L_channel_LAB(img_rgb):
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel_img = lab_img[:, :, 0]
    return l_channel_img


def get_B_channel_LAB(img_rgb):
    lab_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    b_channel_img = lab_img[:, :, 2]
    return b_channel_img


def convert_and_threshold(input_img, thresholds, visu=False):
    '''
    Pipeline for combining thresholds for Sobel and magnitude
    applied on grayscale and S channel (hls colour space) images
    '''

    # 0. Get L channel from input image
    channel_img = get_L_channel_LAB(input_img)

    # 1. Calculate absolute threshold and magnitude threshold for grayscale image
    abs_sobel_bin_img = abs_sobel_thresh(input_img, orient='x',
                                         thresh=(thresholds.grayscale_abs_bottom, thresholds.grayscale_abs_upper))
    mag_thresh_bin_img = mag_thresh(input_img, sobel_kernel=5, mag_thresh=(
    thresholds.grayscale_magnitude_bottom, thresholds.grayscale_magnitude_upper))

    # 2.  Calculate absolute threshold and magnitude threshold for s channel image
    abs_sobel_bin_img_gray = abs_sobel_thresh(channel_img, orient='x', thresh=(
    thresholds.channel_abs_bottom, thresholds.channel_abs_upper))
    mag_thresh_bin_img_gray = mag_thresh(channel_img, sobel_kernel=15, mag_thresh=(
    thresholds.channel_magnitude_bottom, thresholds.channel_magnitude_upper))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(channel_img)
    combined_binary[((abs_sobel_bin_img == 1) | (mag_thresh_bin_img == 1))
                    | ((abs_sobel_bin_img_gray == 1) | (mag_thresh_bin_img_gray == 1))] = 1

    if visu == True:
        # Plotting thresholded images
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(
            ncols=2,
            nrows=3,
            figsize=(20, 10))

        ax1.set_title('Absolute Sobel')
        ax1.imshow(abs_sobel_bin_img, cmap='gray')

        ax2.set_title('Magnitute threshold')
        ax2.imshow(mag_thresh_bin_img, cmap='gray')

        ax3.set_title('Absolute Sobel - L channel (LAB)')
        ax3.imshow(abs_sobel_bin_img_gray, cmap='gray')

        ax4.set_title('Magnitute threshold - L channel (LAB)')
        ax4.imshow(mag_thresh_bin_img_gray, cmap='gray')

        ax5.set_title('L channel (LAB)')
        ax5.imshow(channel_img)

        ax6.set_title('Combined thresholds')
        ax6.imshow(combined_binary, cmap='gray')

        plt.show()

    return combined_binary


def save_binary_image(img, name_dir, img_name):
    directory = '../test_images/' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + img_name
    cv2.imwrite(directory, img.astype('uint8') * 255)


def main():
    '''
    Read and visualize thresholding results
    '''
    images_names = glob.glob('../test_images/undistorted/test*.jpg')
    destination_dir_name = "combined_binary"
    thresholds = Thresholds()  # use default thresholds
    for fname in images_names:
        img = cv2.imread(fname)
        combined_binary = convert_and_threshold(img, thresholds, visu=True)
        save_binary_image(combined_binary, destination_dir_name, ntpath.basename(fname))


def set_up_thresholds_values():
    tune_image_name = '../test_images/undistorted/test4.jpg'
    thresholds = Thresholds()  # use default thresholds
    img = cv2.imread(tune_image_name)
    combined_binary = choose_threshold_value(img)


if __name__ == '__main__':
    main()
    #set_up_thresholds_values()
