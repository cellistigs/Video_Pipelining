## File to take in a numpy sequence of filenames, and then generate augmentation
## data for a particular video.
### This is based on Data_supp file found in DLC_edits.
import imageio
imageio.plugins.ffmpeg.download()
import matplotlib
matplotlib.use('Agg')
from moviepy.editor import VideoFileClip
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np
import os
import math
import sys
sys.path.append(os.getcwd().split('Generating_a_Training_Set')[0])

def attempttomakefolder(foldername):
    if os.path.isdir(foldername):
        print("Folder already exists!")
    else:
        os.mkdir(foldername)

def SelectFramesFinetune(videopath,video,x1,x2,y1,y2,cropping,Task,indices):
    ''' Selecting frames from videos for labeling.'''

    basefolder = 'data-' + Task + '/'
    attempttomakefolder(basefolder)
    print("Loading ", video)
    clip = VideoFileClip(os.path.join(videopath, video))
    print("Duration of video [s], ", clip.duration, "fps, ", clip.fps,
          "Cropped frame dimensions: ", clip.size)

    ####################################################
    # Creating folder with name of experiment and extract random frames
    ####################################################
    folder = video.split('.')[0]+'_supp'
    attempttomakefolder(os.path.join(basefolder,folder))
    indexlength = int(np.ceil(np.log10(clip.duration * clip.fps)))
    # Extract the first frame (not cropped!) - useful for data augmentation
    index = 0

    image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))

    io.imsave(os.path.join(basefolder,folder,"img" + str(index).zfill(indexlength) + ".png"),image)

    if cropping==True:
        #Select ROI of interest by adjusting values in myconfig.py
        clip=clip.crop(y1=y1,y2=y2,x1 = x1,x2=x2)
    print("Extracting frames ...")
    frames2pick = indices

    indexlength = int(np.ceil(np.log10(clip.duration * clip.fps)))
    for index in frames2pick:
        print(index)
        try:
            image = img_as_ubyte(clip.get_frame(index/clip.fps))
            print(clip.get_frame(index/clip.fps))
            io.imsave(os.path.join(basefolder,folder,"img" + str(index).zfill(indexlength) + ".png"),image)
        except FileNotFoundError:
            print("Frame # ", index, " does not exist.")

if __name__ == '__main__':
    cropping = True
    x1 = 330
    x2 = 630
    y1 = 70
    y2 = 480
    Task = 'social_v4'
    videos = ['V118_03182018_cohousing-Camera-1-Animal-1-Segment1cropped_part9.mp4','V118_03182018_cohousing-Camera-1-Animal-1-Segment1cropped_part19.mp4','V118_03182018_cohousing-Camera-1-Animal-1-Segment1cropped_part29.mp4']## Give video here
    videopath = "../DeepLabCut/videos/S3/V118/V119_03182018/"
    all_indices = [[12163,12164,17289,17290,17300,17311,17312,17340,17347,17372,
                    17389,19146,19148,19161,19808,19820,31101,24960,25770,32846,
                    32922,32930,33118,32040,32045,32760,32850],[13183,13184,13192,
                    13194,13196,13270,13271,13275,13290,13314,13319,13357,13416,
                    13417,13440,13448,15696,15480,15481,15482,15856,15879,15870,15893,
                    15890,16170,16171,23850,23851,30180,30181],[1966,1967,3292,1530,
                    1531,1800,1950,1955,1960,3270,3271,3272,3380,3381,3385,3392,3394,3396,
                    3417,3419,3423,5130,5160,5156,5161,6945,6946,6953,6957,6965,6989,6990,
                    6992,7004,14820,14825,14815,14835,17100,22338,23160,23670]]##
    for i,video in enumerate(videos):
        indices = all_indices[i]
        SelectFramesFinetune(videopath,video,x1,x2,y1,y2,cropping,Task,indices)
