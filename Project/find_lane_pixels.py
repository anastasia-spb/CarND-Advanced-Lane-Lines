import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    binary_filtered, histogram = get_hist(binary_warped)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
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

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_filtered.shape[0] - (window + 1) * window_height
        win_y_high = binary_filtered.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

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

    return leftx, lefty, rightx, righty

def fit_polynomial(binary_warped):
   # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return left_fitx, right_fitx, left_fit, right_fit


def filter_hist(histogram, thresh = 50, max_len = 100):
    count = 0
    start = 0
    stop = 0
    result_array = np.array([], dtype='int')
    for i in range(histogram.size):
        if histogram[i] > thresh:
            if count == 0:
                start = i
            count += 1
            if i == (histogram.size - 1) and count > 0:
                stop = start + count
                if count > max_len:
                    arr = np.arange(start, stop, 1, dtype='int')
                    result_array = np.append(result_array, arr)
            continue
        elif count > 0:
            stop = start + count
            if count > max_len:
                arr = np.arange(start, stop, 1, dtype='int')
                result_array = np.append(result_array, arr)
            count = 0
    return result_array


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
    histogram = np.sum(binary_warped[height//2:height,:], axis=0)
    noise = filter_hist(histogram)
    histogram[noise] = 0
    binary_filtered = np.copy(binary_warped)
    binary_filtered[:,noise] = 0

    return binary_filtered, histogram

def test_histogram():
    images_names = glob.glob('../test_images/warped/test*.jpg')
    for fname in images_names:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img = img/255
        visu_histogram(img)

if __name__ == '__main__':
    test_histogram()