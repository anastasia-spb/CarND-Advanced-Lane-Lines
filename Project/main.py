import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration
import frame_pipeline
import os
import ntpath
import process_video

def save_image(img, img_name, name_dir = "output_images"):
    directory = '../' + name_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/' + img_name
    cv2.imwrite(directory, img)

def test_on_test_images():
    images_names = glob.glob('../test_images/test*.jpg')
    for fname in images_names:
        test_on_image(fname)

def test_on_image(fname = '../test_images/test7.jpg'):
    cam_mtx, dist_coeff = camera_calibration.get_calib_parameters()
    img = cv2.imread(fname)
    M, Minv, mask_points, mask_points_inner, result = frame_pipeline.first_frame_pipeline(img, cam_mtx, dist_coeff)
    save_image(result, ntpath.basename(fname))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

def test_process_video():
    video_full_name = "../videos/project_video.mp4"
    cam_mtx, dist_coeff = camera_calibration.get_calib_parameters()
    process_video.process_video(video_full_name, cam_mtx, dist_coeff, output_video_name = "resulting_video.avi")

if __name__ == '__main__':
    test_on_image()
    #test_on_test_images()
    #test_process_video()

