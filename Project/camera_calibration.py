import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def calibrate_camera(images_names):
    nx = 9  # chessboard size
    ny = 6  # chessboard size
    objp = np.zeros((ny * nx, 3), np.float32)
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
    cam_mtx = np.array([[1.15777930e+03, 0.00000000e+00, 6.67111054e+02],
                        [0.00000000e+00, 1.15282291e+03, 3.86128937e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    return cam_mtx

def get_dist_coeff():
    dist_coeff = np.array([[-0.24688775, -0.02373133, -0.00109842,  0.00035108, -0.00258571]])
    return dist_coeff

def get_calib_parameters(recalibrate = False):
    if recalibrate == True:
        images_names = glob.glob('../camera_cal/calibration*.jpg')  # < Make a list of calibration images
        cam_mtx, dist_coeff = calibrate_camera(images_names)
        return cam_mtx, dist_coeff
    else:
        return get_cam_matrix(), get_dist_coeff()

def main():
    cam_mtx, dist_coeff = get_calib_parameters()

if __name__ == '__main__':
    main()
