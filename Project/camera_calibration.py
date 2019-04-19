import numpy as np
import cv2
import os
import glob
import ntpath
import matplotlib.pyplot as plt

def calibrate_camera(images_names):
    '''
    Calibrate camera on chessboard images
    :param images_names: list of images names on which calibration shall be done
    :return: camera matrix, distortion coefficients
    '''
    nx = 9  # chessboard size
    ny = 6  # chessboard size
    objp = np.zeros((ny * nx, 3), np.float32)
    # create a multi-dimensional mesh grid, transpose it, reshape
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for fname in images_names:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    # perform camera calibration
    [ret, mtx, dist, rvecs, tvecs] = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]), None, None)

    # return camera matrix and distortion coefficients
    return mtx, dist


def get_cam_matrix():
    '''
    an energy-independent memory of the vehicle
    :return: camera matrix
    '''
    cam_mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
                        [0.00000000e+00, 1.15282291e+03, 3.86128937e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return cam_mtx

def get_dist_coeff():
    '''
    an energy-independent memory of the vehicle
    :return: distortion coefficients
    '''
    dist_coeff = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])
    return dist_coeff

def get_calib_parameters(recalibrate = False):
    '''
    Get calibration parameters such as camera matrix and distortion coefficients
    :param recalibrate: If true, calibrate camera on images from '../camera_cal' folder. Otherwise return stored values.
    :return: camera matrix, distortion coefficients
    '''
    if recalibrate == True:
        images_names = glob.glob('../camera_cal/calibration*.jpg')  # < Make a list of calibration images
        cam_mtx, dist_coeff = calibrate_camera(images_names)
        print(cam_mtx, dist_coeff)
        return cam_mtx, dist_coeff
    else:
        return get_cam_matrix(), get_dist_coeff()


def save_image(img, name_dir, img_name):
    directory = '../test_images/' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + img_name
    cv2.imwrite(directory, img)

def calibrate_and_store_test_images(dir_name = "undistorted"):
    '''
    Undistort all test images and store them into folder
    '''
    images_names = glob.glob('../test_images/test*.jpg')
    for fname in images_names:
        img = cv2.imread(fname)
        img_undist = cv2.undistort(img, get_cam_matrix(), get_dist_coeff(), None, get_cam_matrix())
        save_image(img_undist, dir_name, ntpath.basename(fname))


def main():
    cam_mtx, dist_coeff = get_calib_parameters(recalibrate = False)

if __name__ == '__main__':
    main()
    # uncomment following line if you want to refresh all undistorted test images images
    #calibrate_and_store_test_images()
