## Functions and script to determine if videos have motion in them or not.
## Assumes that all visible segments have been analyzed with deeplabcut, will
## complain otherwise.
import os
from moviepy.editor import VideoFileClip
import numpy as np
import sys
import subprocess
import re
import gc
from Social_Dataset_Class import social_dataset
import joblib

def motion_detection(cwd = None):
    motiondict = {}
    length = 1200
    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    print(cwd)
    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    all_sub = all_sub
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        # Find all reference videos:
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        print(videos)
        for video in videos:
            # Find videos that have been analyzed:
            # We need to identify the specific video that we are building from.
            ident =  video.split('.')[0]+'cropped_'+'part'
            print(ident)
            chopped = [part for part in files if ident in part.split('.')[0] and part.split('.')[1] == 'mp4']
            print(chopped)
            if len([part for part in files if ident in part.split('.')[0] and part.split('.')[1] == 'mp4']):
                ## For each part load in the appropriate dataset
                for part in chopped:
                    try:
                        video_id = part.split('.mp4')[0]
                        data_id = video_id + 'DeepCut_resnet50_social_v3Nov2shuffle1_1030000.h5'
                        video_dataset = social_dataset(cwd+'/'+sub+'/'+data_id)
                        output = video_dataset.motion_detector()
                        print(output)
                        motiondict[data_id] = output

                    except OSError as error:
                        print(error)
            else:
                print('video: '+part +'has not been cropped')
    return motiondict

def save_tracking(cwd = None):
    motiondict = joblib.load('motiondict')
    length = 1200
    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    print(cwd)
    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    all_sub = all_sub
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        # Find all reference videos:
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        print(videos)
        for video in videos:
            # Find videos that have been analyzed:
            # We need to identify the specific video that we are building from.
            ident =  video.split('.')[0]+'cropped_'+'part'
            print(ident)
            chopped = [part for part in files if ident in part.split('.')[0] and part.split('.')[1] == 'mp4']
            print(chopped)
            if len([part for part in files if ident in part.split('.')[0] and part.split('.')[1] == 'mp4']):
                ## For each part load in the appropriate dataset
                for part in chopped:
                    try:
                        video_id = part.split('.mp4')[0]
                        data_id = video_id + 'DeepCut_resnet50_social_v3Nov2shuffle1_1030000.h5'
                        # Check if the data is static, then don't bother:
                        if np.all(motiondict[data_id] == 0):
                            print('skipping static segment')
                        # Otherwise filter the data appropriately:
                        else:
                            video_dataset = social_dataset(cwd+'/'+sub+'/'+data_id)
                            video_dataset.filter_check_replaces([0,1,2,3,4,5,6,7,8,9])
                            joblib.dump(video_dataset,cwd+'/'+sub+'/'+video_id+'datastruct')


                    except OSError as error:
                        print(error)
            else:
                print('video: '+part +'has not been cropped')


# def nest_detection()
if __name__ == "__main__":
    cwd = sys.argv[1]
    save_tracking(cwd)

    # joblib.dump(moving,"motiondict")
