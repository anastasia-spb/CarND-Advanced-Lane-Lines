# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
import matplotlib.pyplot as plt

import frame_pipeline

def process_video(video_full_name, cam_mtx, dist_coeff, output_video_name = "resulting_video.avi"):
    frames = []
    is_first = True
    M = np.array([])
    Minv = np.array([])
    mask_points = []

    processed_frames_count = 0
    vidcap = cv2.VideoCapture(video_full_name)
    success, image = vidcap.read()
    while success:
        processed_frames_count += 1
        print("Frame {:d}".format(processed_frames_count))
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 12
    video = cv2.VideoWriter(output_video_name, fourcc, float(fps), (width, height))
    for frame in frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()