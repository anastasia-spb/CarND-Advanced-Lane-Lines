import numpy as np
import cv2
import os
import glob
import ntpath
import matplotlib.pyplot as plt

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
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
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
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

def get_S_channel(img_rgb):
    hls_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    s_channel_img = hls_img[:,:,2]
    return s_channel_img

def get_V_channel_Luv(img_rgb):
    luv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Luv)
    v_channel_img = luv_img[:,:,2]
    return v_channel_img


def convert_and_threshold(input_img, visu = False):
    # 0. Get S channel from input image
    s_channel_img = get_S_channel(input_img)

    # 1. Calculate absolute threshold and magnitude threshold for grayscale image
    abs_sobel_bin_img = abs_sobel_thresh(input_img, orient='x', thresh=(20, 100))
    mag_thresh_bin_img = mag_thresh(input_img, sobel_kernel=5, mag_thresh=(20, 100))

    # 2.  Calculate absolute threshold and magnitude threshold for s channel image
    abs_sobel_bin_img_gray = abs_sobel_thresh(s_channel_img, orient='x', thresh=(50, 100))
    mag_thresh_bin_img_gray = mag_thresh(s_channel_img, sobel_kernel=15, mag_thresh=(30, 100))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_channel_img)
    combined_binary[((abs_sobel_bin_img == 1) & (mag_thresh_bin_img == 1))
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

        ax3.set_title('Absolute Sobel - S channel')
        ax3.imshow(abs_sobel_bin_img_gray, cmap='gray')

        ax4.set_title('Magnitute threshold - S channel')
        ax4.imshow(mag_thresh_bin_img_gray, cmap='gray')

        ax5.set_title('S channel')
        ax5.imshow(s_channel_img)

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
    for fname in images_names:
        img = cv2.imread(fname)
        combined_binary = convert_and_threshold(img, visu = False)
        save_binary_image(combined_binary, destination_dir_name, ntpath.basename(fname))

if __name__ == '__main__':
    main()