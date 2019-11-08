########################################################################
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipdb
########################################################################
def colour_segmentation(img, object_key_word, threshold_factor, give_pic = False):

    """
    Given an image and colour, return an image that is segmented based on
    the regions that match the colour.

    Assume that the input image has shape (84, 84, 3)
    Outputs an image of shape(84, 84, 3)
    """

    object_colour_dict = {
        'globe' : np.array([ 81, 156, 255]),
        'time' : np.array([0, 255, 0]),
        'green_door' : np.array([61, 121, 70]),
        'next_level' : np.array([249 , 232,  24])
    }

    #get the std of each image channel and determine acceptable threshold
    channel_image = img.reshape(img.shape[0] * img.shape[1], 3)
    threshold = channel_image.std(axis = 0) * threshold_factor

    #get the matching pixels
    subtracted_mean_image = np.abs(channel_image - object_colour_dict[object_key_word])
    matches = np.where(np.where(subtracted_mean_image < threshold, 1, 0).sum(axis = 1) == 3, 1, 0)
    matches = matches.reshape(img.shape[0], img.shape[1])
    if object_key_word == 'globe' and matches.sum() > 0:
        ipdb.set_trace()
    #filter for matches
    if give_pic: #either return a new image or just the matches
        new_image = img.copy()
        for d in [0,1,2]:
            new_image[:,:,d] = matches * img[:,:,d]
        return new_image
    else:
        return matches
########################################################################
def colour_scoring(img, previous_time = 0.1):
    """
    Based on the colour matching, give the agent an reward.
    """

    weights = { #the smaller the weight the more important it will be
        'globe': 100000,
        'green_door': 50000,
        'next_level': 10000,
        'time': 100000
    }

    #retrieve the matching colours from the images
    scores = np.array([])
    matches = {}
    matches['globe'] = colour_segmentation(img, 'globe', 0.5, give_pic = False).sum()
    matches['green_door'] = colour_segmentation(img, 'green_door', 0.25, give_pic = False).sum()
    matches['next_level'] = colour_segmentation(img, 'next_level', 0.5, give_pic = False).sum()
    matches['time'] = colour_segmentation(img, 'time', 0.5, give_pic = False).sum()

    #if the time has gone up we can give reward otherwise zero
    matches['time_diff'] = 0
    if matches['time'] + 10 <= previous_time or matches['time'] == 100:
        matches['time_diff'] = 0
    else:
        matches['time_diff'] = (matches['time'] - previous_time)/weights['time']
    if matches['time_diff'] >0:
        ipdb.set_trace()

    #weight the matches
    scores = np.array([
        matches['globe']/weights['globe'],
        matches['green_door']/weights['green_door'],
        matches['next_level']/weights['next_level'],
        matches['time_diff'],
        ])

    return matches['time'], scores
########################################################################
