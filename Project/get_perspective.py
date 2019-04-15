import numpy as np
import cv2
import glob
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
    # upper left point
    x = points[0][0] - offset_top
    y = points[0][1]
    points_list.append([x, y])
    # upper right point
    x = points[1][0] + offset_top
    y = points[0][1]  # on the same level as upper left point
    points_list.append([x, y])
    # bottom right point
    x = points[2][0] + offset_bottom
    y = points[2][1]
    points_list.append([x, y])
    # bottom left point
    x = points[3][0] - offset_bottom
    y = points[2][1]
    points_list.append([x, y])

    src = np.float32(points_list)
    return src


def set_destination_points(input_img, offset_x_left=0, offset_x_right=0, offset_y=400):
    [x, y, z] = input_img.shape
    dst = np.float32(
        [[offset_x_left, offset_y], [x - offset_x_right, offset_y], [x - offset_x_right, y], [offset_x_left, y]])
    return dst


def get_perpective_matrix():
    M = np.array([[-5.19791054e-01, -8.21071218e-01, 6.76450118e+02],
                  [-2.16134672e-16, -4.15133902e+00, 2.00156342e+03],
                  [-5.63756296e-19, -2.38651413e-03, 1.00000000e+00]])
    return M

def calculate_perpective_matrix(input_img, calculate_matrix=False):
    if calculate_matrix == True:
        points = choose_points(input_img)
        src = align_manually_chosen_points(points)
        dst = set_destination_points(input_img)
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        M = get_perpective_matrix()
    return M


def main():
    img_ref = cv2.imread('../test_images/straight_lines1.jpg')
    M = calculate_perpective_matrix(img_ref)
    print(M)
    img_size = (img_ref.shape[1], img_ref.shape[0])
    warped = cv2.warpPerspective(img_ref, M, img_size, flags=cv2.INTER_LINEAR)
    # Visualize
    fig, ((ax1, ax2)) = plt.subplots(ncols=2, nrows=1, figsize=(20, 10))
    ax1.set_title('Original image')
    ax1.imshow(img_ref)

    lineThickness = 2
    x = 50
    cv2.line(warped, (x, 0), (x, img_ref.shape[0]), (0, 255, 0), lineThickness)

    ax2.set_title('Bird eye view')
    ax2.imshow(warped)

    plt.show()

if __name__ == '__main__':
    main()
