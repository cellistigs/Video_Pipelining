## i script to search through a set of videos organized in the relevant file structure,
## and retrieve one frame from each. It will involve functions that take as arguments
## a cropping frame, and a rotation if necessary. The rotation will be applied first.

## assume that we are in a folder containing a single pair of animals. Assume furhter
## that subfolders contain videos as avis. (may be more than one per video)

# Save one frame of each video.
import os
from moviepy.editor import VideoFileClip
import numpy as np
import sys
import subprocess
import re
def inspect_videos(cwd = None):

    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:

    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        for video in videos:
            clip = VideoFileClip(cwd+'/'+sub+'/'+video)
            clip.save_frame(cwd+'/'+sub+'/'+video.split('.')[0]+'testframe.png')
            with open(cwd+'/'+sub+'/'+video.split('.')[0]+'config.py','w+') as f:
                coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                for coord in coords:
                    f.write(coord)

def crop_videos(cwd = None):

    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    print(cwd)
    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        # Only look at videos
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        print(videos)
        for video in videos:
            ## Load in cropped version of
            print('loading ' +video)
            clip = VideoFileClip(cwd+'/'+sub+'/'+video)
            try:
                with open(cwd+'/'+sub+'/'+video.split('.')[0]+'config.py','r+') as f:
                    coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                    intcoords = []
                    for coord in range(len(coords)):
                        coords[coord] = f.readline()
                        nums = re.findall('\d+',coords[coord])[1]
                        intcoords.append(nums)

                print(video.split('.')[0]+'cropped.avi')
                cropped = clip.crop(x1 = intcoords[0],y1 = intcoords[1],x2 = intcoords[2],y2 = intcoords[3])
                ## We want to split our video into one hour segments.
                ## First get the duration in seconds:
                seconds = cropped.duration
                segments = np.ceil(seconds/1200).astype(int) # rounds up to give the number of distinct segments we need
                for segment in range(segments):
                    # Ensures that the last clip is the right length
                    print("producing segment "+str(segment) + 'of ' + str(segments))
                    if segment == segments-1:
                        endseg = -1
                    else:
                        endseg = 1200*(segment+1)
                    cropped_cutout = cropped.subclip(t_start = segment*1200,t_end = endseg)
                    cropped_cutout.write_videofile(cwd+'/'+sub+'/'+video.split('.')[0]+'cropped_'+'part' +str(segment)+ '.mp4',codec = 'mpeg4',bitrate = "1500k",threads = 2)

            except OSError as e:

                 print(e.errno)
                 print('configuration not loaded')


def crop_videos_debug(cwd = None):

    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    print(cwd)
    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        # Only look at videos
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        print(videos)
        for video in videos:
            ## Load in cropped version of
            print('loading ' +video)
            clip = VideoFileClip(cwd+'/'+sub+'/'+video)

            with open(cwd+'/'+sub+'/'+video.split('.')[0]+'config.py','r+') as f:
                coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                intcoords = []
                for coord in range(len(coords)):
                    coords[coord] = f.readline()
                    nums = re.findall('\d+',coords[coord])[1]
                    intcoords.append(nums)

            print(video.split('.')[0]+'cropped.avi')
            cropped = clip.crop(x1 = intcoords[0],y1 = intcoords[1],x2 = intcoords[2],y2 = intcoords[3])
            cropped_cutout = cropped.subclip(t_start = 0,t_end = 100)
            #print(cropped_cutout.duration,cropped_cutout.fps)
            cropped_cutout.write_videofile(cwd+'/'+sub+'/'+video.split('.')[0]+'cropped.avi',codec = 'libx264',preset = 'fast',threads = 2)


def compress_videos(cwd = None):

    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:

    all_sub = next(os.walk(cwd))[1]
    print(all_sub)
    for sub in all_sub:
        files = os.listdir(cwd+'/'+sub)
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        for video in videos:
            print('working on ' +video)
            command = 'ffmpeg','-y','-i',cwd+'/'+sub+'/'+video, '-vcodec', 'libx264', '-crf', '23', cwd+'/'+sub+'/'+video.split('.')[0]+'compressed.avi'
            inputcommand = command
            program = 'ffmpeg'
            print(program,inputcommand)
            s = subprocess.call(command)
            if s == 0:
                print('compressing '+cwd+'/'+sub+'/'+video)
            else:
                print('could not compress '+cwd+'/'+sub+'/'+video)

if __name__ == "__main__":
    cwd = sys.argv[1]
    crop_videos(cwd)
