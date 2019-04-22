import numpy as np
import cv2
import os
import glob
import ntpath
import matplotlib.pyplot as plt


class ParamsStruct():
    def __init__(self, count=0, window_name="", img=None, points=[], draw_help_lines = True):
        self.count = count
        self.window_name = window_name
        self.img = img
        self.points = points
        self.draw_help_lines = draw_help_lines


def mouseCB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param.img, (x, y), 3, (0, 0, 255), 3)  # BGR
        if param.count == 0 and param.draw_help_lines == True:
            cv2.line(param.img, (x, y), (param.img.shape[1], y), (0, 0, 255), 3)
        if param.count == 2 and param.draw_help_lines == True:
            cv2.line(param.img, (0, y), (x, y), (0, 0, 255), 3)
        param.count += 1
        param.points.append([x, y])
        cv2.imshow(param.window_name, param.img)


def choose_points(img_input, draw_help_lines = True):
    """
    Choose four input points for calculation matrix
    for perspective transformation and then press 'q'
    for quit
    1. Choose upper left points
    2. Choose upper right point
    3. Choose bottom right point
    4. Choose bottom left point
    Important observations: choose bottom points
    as much as possible close to the image bottom
    """

    copy_img = img_input.copy()
    window_name = "Choose source points"
    points = []
    param = ParamsStruct(0, window_name, copy_img, points, draw_help_lines)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, copy_img)
    cv2.setMouseCallback(window_name, mouseCB, param)

    while True:
        ip = cv2.waitKey(0) & 0xFF
        if ip == ord('q'):
            break

    cv2.destroyAllWindows()
    return param.points


def align_manually_chosen_points(points, offset_top=0, offset_bottom=0):
    # We suppose that lines are parallel
    points_list = []
    dist = points[0][0] - points[3][0]
    # upper left point
    x = points[0][0] - offset_top
    y = points[0][1]
    points_list.append([x, y])
    # upper right point
    x = points[1][0] + offset_top
    y = points[0][1]  # on the same level as upper left point
    points_list.append([x, y])
    # bottom right point
    x = points[1][0] + dist + offset_bottom
    y = points[2][1]
    points_list.append([x, y])
    # bottom left point
    x = points[3][0] - offset_bottom
    y = points[2][1]
    points_list.append([x, y])

    src = np.float32(points_list)
    return src, points_list


def roi(img, vertices):
    # blank mask
    mask = np.zeros_like(img)
    # filling pixels inside the polygon defined by vertices with the fill color
    cv2.fillPoly(mask, vertices, 255)
    # returning the image with all pixels
    masked_background = cv2.bitwise_or(img, mask)
    return masked_background


def plot_mask_on_image(src_img, points_list):
    vertices = np.array([points_list], dtype=np.int32)
    masked_img = roi(src_img, vertices)
    return masked_img


def set_destination_points(input_img, offset_x=300, offset_y=0):
    [height, width, z] = input_img.shape
    dst = np.float32(
        [[offset_x, offset_y], [width - offset_x, offset_y], [width - offset_x, height-offset_y], [offset_x, height-offset_y]])
    return dst


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def mask_image(input_img):
    masked_img = np.copy(input_img)
    mask_points = choose_points(input_img, draw_help_lines = False)
    vertices_mask = np.array([mask_points], dtype=np.int32)
    masked_img = region_of_interest(input_img, vertices_mask)
    return masked_img


def calculate_perpective_matrix(input_img):
    points = choose_points(input_img)
    src, points_list = align_manually_chosen_points(points)
    result = plot_mask_on_image(input_img, points_list)
    dst = set_destination_points(input_img)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    return M, Minv, result


def visualize(img_ref, warped):
    # Visualize
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax1.set_title('Original image')
    ax1.imshow(img_ref)

    img_ref_with_line = np.copy(warped)
    lineThickness = 2
    x = 50
    cv2.line(img_ref_with_line, (x, 0), (x, img_ref.shape[0]), (0, 255, 0), lineThickness)

    ax2.set_title('Transformed image')
    ax2.imshow(img_ref_with_line)

    plt.show()

    return


def save_binary_image(img, M, Minv, name_dir, img_name):
    directory = '../test_images/' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    img_name_no_ext = os.path.splitext(img_name)[0]
    directory_img = directory + '/' + img_name_no_ext + '.npy'
    directory_M = directory + '/M_' + img_name_no_ext + '.npy'
    directory_M_inv = directory + '/Minv_' + img_name_no_ext + '.npy'
    np.save(directory_img, img)
    np.save(directory_M, M)
    np.save(directory_M_inv, Minv)


def load_and_visu_stored_arrays():
    arrays_names = glob.glob('../test_images/warped/test*.npy')
    for fname in arrays_names:
        img = np.load(fname)
        plt.imshow(img)
        plt.show()


def main():
    show_results = False
    if show_results == False:
        images_names = glob.glob('../test_images/combined_binary/test*.jpg')
        destination_dir_name = "warped"
        for fname in images_names:
            img = cv2.imread(fname)
            M, Minv, img_with_marked_area = calculate_perpective_matrix(img)
            masked_img = mask_image(img)
            img_size = (masked_img.shape[1], masked_img.shape[0])
            warped = cv2.warpPerspective(masked_img, M, img_size, flags=cv2.INTER_LINEAR)
            save_binary_image(warped, M, Minv, destination_dir_name, ntpath.basename(fname))
            visualize(img_with_marked_area, warped)
    else:
        load_and_visu_stored_arrays()


if __name__ == '__main__':
    main()
