import numpy as np
import cv2
import os
import glob
import ntpath
import matplotlib.pyplot as plt


class ParamsStruct():
    def __init__(self, count=0, window_name="", img=None, points=[]):
        self.count = count
        self.window_name = window_name
        self.img = img
        self.points = points


def mouseCB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param.img, (x, y), 3, (0, 0, 255), 3)  # BGR
        if param.count == 0:
            cv2.line(param.img, (x, y), (param.img.shape[1], y), (0, 0, 255), 3)
        if param.count == 2:
            cv2.line(param.img, (0, y), (x, y), (0, 0, 255), 3)
        param.count += 1
        param.points.append([x, y])
        cv2.imshow(param.window_name, param.img)


def choose_points(img_input):
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
    param = ParamsStruct(0, window_name, copy_img, points)
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

def roi(img,vertices):
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


def set_destination_points(input_img, offset_x=100, offset_y=300):
    [x, y, z] = input_img.shape
    dst = np.float32(
        [[offset_x, offset_y], [x - offset_x, offset_y], [x - offset_x, y], [offset_x, y]])
    return dst


def get_perpective_matrix():
    M = np.array([[-4.13308420e-01, -8.37005327e-01, 6.15526118e+02],
                  [-1.64763533e-15, -3.83969657e+00, 1.84070725e+03],
                  [-1.33349842e-18, -2.38795772e-03, 1.00000000e+00]])
    return M


def calculate_perpective_matrix(input_img, calculate_matrix=False):
    if calculate_matrix == True:
        points = choose_points(input_img)
        src, points_list = align_manually_chosen_points(points)
        result = plot_mask_on_image(input_img, points_list)
        dst = set_destination_points(input_img)
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = get_perpective_matrix()
        result = input_img
    return M, result


def visualize(img_ref, warped):
    # Visualize
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax1.set_title('Original image')
    ax1.imshow(img_ref)

    lineThickness = 2
    x = 50
    cv2.line(warped, (x, 0), (x, img_ref.shape[0]), (0, 255, 0), lineThickness)

    ax2.set_title('Transformed image')
    ax2.imshow(warped)

    plt.show()

    return

def save_binary_image(img, name_dir, img_name):
    directory = '../test_images/' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + img_name
    cv2.imwrite(directory, img.astype('uint8') * 255)

def main():
    images_names = glob.glob('../test_images/combined_binary/test*.jpg')
    destination_dir_name = "warped"
    for fname in images_names:
        img = cv2.imread(fname)
        M, masked_img = calculate_perpective_matrix(img, calculate_matrix = True)
        img_size = (img.shape[1], img.shape[0])
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
        visualize(masked_img, warped)
        save_binary_image(warped, destination_dir_name, ntpath.basename(fname))


if __name__ == '__main__':
    main()
