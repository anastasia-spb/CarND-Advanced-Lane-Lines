# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
import numpy as np
import matplotlib.pyplot as plt

import frame_pipeline

def process_video(video_full_name, cam_mtx, dist_coeff):
    frames = []
    is_first = True
    M = np.array([])
    Minv = np.array([])
    mask_points = []

    vidcap = cv2.VideoCapture(video_full_name)
    success, image = vidcap.read()
    while success:
        if is_first == True:
            M, Minv, mask_points, result = frame_pipeline.first_frame_pipeline(image, cam_mtx, dist_coeff)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.show()
            is_first = False
        else:
            result = frame_pipeline.frame_pipeline(image, M, Minv, cam_mtx, dist_coeff, mask_points)
        frames.append(result)
        success, image = vidcap.read()
    # write frames into video file
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter("new.mp4", 0, 1, (width, height))
    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()