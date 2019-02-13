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
import gc
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
            try:
                config = open(cwd+'/'+sub+'/'+video.split('.')[0]+'config.py')
            except:
                clip.save_frame(cwd+'/'+sub+'/'+video.split('.')[0]+'testframe.png')
                with open(cwd+'/'+sub+'/'+video.split('.')[0]+'config.py','w+') as f:
                    coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                    for coord in coords:
                        f.write(coord)

def crop_videos(cwd = None):
    length = 1200
    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    print(cwd)
    all_sub = next(os.walk(cwd))[1]
    print(all_sub[5:])
    all_sub = all_sub
    for sub in all_sub[5:]:
        files = os.listdir(cwd+'/'+sub)
        # Only look at videos
        videos = [video for video in files if video.split('.')[-1] == 'avi']
        print(videos)
        for video in videos:
            ## Load in cropped version of
            print('loading ' +video)
            #clip = VideoFileClip(cwd+'/'+sub+'/'+video)
            print(cwd,sub)
            clip = VideoFileClip(cwd+'/'+sub+"/"+video)
            print('IT LOAEDED')
            try:
                with open(cwd+'/'+sub+"/"+video.split('.')[0]+'config.py','r+') as f:
                    coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                    intcoords = []
                    for coord in range(len(coords)):
                        coords[coord] = f.readline()
                        nums = re.findall('\d+',coords[coord])[1]
                        intcoords.append(nums)

                print(video.split('.')[0]+'cropped.avi')
                cropped = clip.crop(x1 = intcoords[0],y1 = intcoords[1],x2 = intcoords[2],y2 = intcoords[3])
                ## We want to split our video into manageable segments.
                ## Account for the case that our video analysis failed somewhere in the middle:
                # We want to be able to extract out the things that have been done so far:

                ## First get the duration in seconds:
                seconds = cropped.duration
                # If analysis has been found:
                # We need to identify the specific video that we are building from. 

                ident =  video.split('.')[0]+'cropped_'+'part'
                print(files,ident)
                print([part for part in files if ident in part.split('.')[0]])
                if len([part for part in files if ident in part.split('.')[0]]):
                    done = [int(re.findall('\d+',part.split('.')[0])[-1]) for part in files if ident in part.split('.')[0]]
                    presegs = range(np.ceil(seconds/length).astype(int))
                    print(done)
                    print(presegs)
                    segments = [segment for segment in presegs if segment not in done]
                    print(segments)

                else:
                    segments = range(np.ceil(seconds/length).astype(int)) # rounds up to give the number of distinct segments we need
                    presegs = segments
                for segment in segments:
                    try:
                        # Ensures that the last clip is the right length
                        print("producing segment "+str(segment) + 'of ' + str(segments))
                        if segment == presegs[-1]:

                            endseg = -1
                        else:
                            endseg = length*(segment+1)
                        cropped_cutout = cropped.subclip(t_start = segment*length,t_end = endseg)
                        cropped_cutout.write_videofile(cwd+'/'+sub+'/'+video.split('.')[0]+'cropped_'+'part' +str(segment)+ '.mp4',codec = 'mpeg4',bitrate = "1500k",threads = 2)
                    except OSError as e:
                        print('segment not viable')
                    gc.collect()
            except OSError as e:

                 print(e.errno)
                 print('configuration not loaded')


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
