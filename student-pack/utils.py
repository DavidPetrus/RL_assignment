import cv2
import os
import numpy as np
from PIL import Image
from random import random
import ipdb

"""
Some utility functions that will assist with training and rendering.
"""

#------------------------------------------------------------------------------
def im2vid(images, video_name = 'video'):

    """
    Take a generator object and converts the images to a video.
    In your RL agent at (S, A, R, S', A') you pass S' to this and it will make a video.
    The reason why I use a generator is because it is much more memory efficient.
    """

    # Arguments
    ext = 'png'
    output = video_name + '.mp4'
    fps = 10

    # Determine the width and height from the first image
    height, width, channels = (168, 168, 3)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    #write the video
    for frame in images:
        out.write(frame) # Write out frame to video

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
#------------------------------------------------------------------------------
def change_resolution(obs, info):

    """
    Merely used for making a video.

    Takes a retro observation and an info
    dictionary and produces a higher resolution
    observation with the retro features tacked on.
    """

    res = (info['brain_info'].visual_observations[0][0] * 255).astype(np.uint8)

    return res
#------------------------------------------------------------------------------
