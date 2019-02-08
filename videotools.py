import moviepy
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import medfilt

## Normalize frame for visualization
def normframe(frame):
    minv,maxv = np.min(frame),np.max(frame)
    normed = 255./(maxv-minv)*(frame-minv)
    return normed.astype(int)
## Get difference between two frames, given in frame number, not seconds
def getdiff(clip,f0,f1):
    frame0 = clip.get_frame(float(f0)/clip.fps).astype(int)
    frame1 = clip.get_frame(float(f1)/clip.fps).astype(int)
    difference = (frame1-frame0)
    return difference
## Given a clip, frame number and template, corrects that frame according to the template.
def correctframe(clip,frame_nb,template,correctionscale = 2.):
    frame = clip.get_frame(frame_nb/float(clip.fps))
    if frame_nb %2 == 0:
        a = template
    else:
        a = 255-template
    corrected = (frame-((a)/correctionscale))
    normed = normframe(corrected)
    return normed

## We want to analyze these automatically. This involves:
# 1: detect massive outliers in frame differences.
def motion_energy(clip):
    ## get duration in frames:
    framecount = clip.duration*clip.fps
    total_energy = np.zeros(int(framecount-1))
    for i in tqdm(range(int(framecount-1))):
        energy = np.sum(np.square(getdiff(clip,i,i+1)))
        total_energy[i] = energy
    return total_energy

def motion_energy_frame(clip,ref = 0):
    ## get duration in frames:
    framecount = clip.duration*clip.fps
    total_energy = np.zeros(int(framecount-1))
    ref = clip.get_frame(ref)
    for i in tqdm(range(int(framecount-1))):
        comp = clip.get_frame(i/clip.fps)
        energy = np.sum(np.square(ref-comp))
        total_energy[i] = energy
    return total_energy

## These videos are of a standardized length of 1200s*30fps frames. Lets see if we can get away
## with creating n templates to subtract from where n divides the video into separate chunks.

def segment(energy,uthresh,lthresh,n,clipduration):

    ## Isolate out areas that have more or less variance, signpost them for segmentation:
    uppers = np.where(energy>uthresh)[0]
    lowers = np.where(energy<lthresh)[0]

    all_out = np.sort(np.concatenate((uppers,lowers)))

    bounds = all_out[np.where(np.diff(all_out)>1)]

    # segment into n different chunks, plus these arbitrary bounds.

    regbounds = np.arange(clipduration)[n::n]

    allbounds = np.sort(np.concatenate((regbounds,bounds,np.array([clipduration]))))

    return allbounds

## Internal function called by clip filterer to create template
def template(clip,lbound,ubound,m):
    ## see if you have enough points to be greater than requested sample count than m:
    segl = ubound-lbound
    possible_indices = np.arange(segl) + lbound
    if m > segl:
        indices = possible_indices
        m = segl
    ## otherwise randomly sample
    else:
        indices = np.sort(np.random.choice(possible_indices,size = m,replace = False))

    # This can be used to construct a background
    # Initialize frame
    frames_shape = np.zeros(4)
    frames_shape[1:] = clip.get_frame(0).shape
    frames_shape[0] = m
    a_init = np.zeros(frames_shape.astype(int))
    a = a_init
    bigind = 0
#     for i,indexi in tqdm(enumerate(indices)):
    for j,indexj in enumerate(indices):
        ## If the jth frame is EVEN, make the ith frame ODD, and vice versa
#         use_indexi = indexi+1*(1-(indexi-indexj)%2)
        use_indexi = indexj+1
        val = getdiff(clip,indexj,use_indexi)

        ## If the jth frame is EVEN, flip the value
        if indexj % 2 == 0:
            val = -val


        a_init[bigind,:,:,:] = val

        bigind+=1

    template = np.median(a_init,axis = 0)

    return(template)

## Internal function used to correct frames as a method of the clip.
def correctframe_inloop(gf,t,fps,bound,template,correctionscale = 2.):

    frame = gf(t/fps)

    if t %2 == 0:
        a = template
        corrected = (frame-((a)/correctionscale))

    else:
        a = -template
        corrected = (frame-((a)/correctionscale))


    normed = np.clip(corrected,0,255)

    return normed

## Improved version, to prevent creating subclips.
def correctframe_inloop_v2(gf,t,fps,templates,indices,correctionscale = 2.):

    temp_ind = indices[t]
    template = templates[temp_ind]
    frame = gf(t/fps)

    if t %2 == 0:
        a = template
        corrected = (frame-((a)/correctionscale))

    else:
        a = -template
        corrected = (frame-((a)/correctionscale))


    normed = np.clip(corrected,0,255)

    return normed


##### We will process the clip as n+ independent subclips, to keep the movie framework simple:
def clip_filterer(clip,energy,m,n):
    ## Upper and lower thresholds currently determined by inspection:
    uthresh = 2.2e8
    lthresh = 1.1e8
    clipduration = clip.fps*clip.duration
    ## Segment
    bounds = segment(m,uthresh,lthresh,n,clipduration)
    ## Now for each set of bounds:
    all_subclips = []
    all_templates = []
    for i in tqdm(range(len(bounds))[:20]):

        lbound = bounds[i]
        if i == len(bounds):
            ubound = clipduration
        else:
            ubound = bounds[i+1]
        ## Now calculate a template:
        segtemplate = template(clip,int(lbound),int(ubound),m)
        ## Determine the part of the clip to which this should be applied:
        segclip = clip.subclip(lbound/float(clip.fps),(ubound+1)/float(clip.fps))
        framefunc = lambda gf,t: correctframe_inloop(gf,int(round(t*clip.fps)),segclip.fps,lbound,segtemplate)

        segfiltered = segclip.fl(framefunc)

        all_subclips.append(segfiltered)
        all_templates.append(segtemplate)
    return all_subclips,all_templates

####### Clip Filter v2 to deal with frame inconsistencies:
##### We will process the clip as n+ independent subclips, to keep the movie framework simple:
##### Note difference: this version will return a lambda function that can be applied to the clip
#####
def clip_filter_v2(clip,energy,m,n):
    ## Upper and lower thresholds currently determined by inspection:
    uthresh = 2.2e8
    lthresh = 1.1e8
    clipduration = int(clip.fps*clip.duration)
    ## Segment
    bounds = segment(energy,uthresh,lthresh,n,clipduration)
    ## Now for each set of bounds:
    all_subclips = []
    all_templates = []
    lbound = 0
    for i in tqdm(range(len(bounds))):

        ubound = bounds[i]

        ## Now calculate a template:
        segtemplate = template(clip,int(lbound),int(ubound),m)

        all_templates.append(segtemplate)

        lbound = ubound

    ## Make an index set that tells each frame which template to use:
    bounds_crossed = 0
    temp_index = np.zeros((clipduration)).astype(int)
    for i in range(clipduration):
        if bounds_crossed == len(bounds):
            temp_index[i] = len(bounds)
        else:
            if i > bounds[bounds_crossed]:
                bounds_crossed +=1
            temp_index[i] = bounds_crossed
    framefunc = lambda gf,t: correctframe_inloop_v2(gf,int(round(t*clip.fps)),clip.fps,all_templates,temp_index)
    subcliped = clip.subclip(t_end = clipduration/clip.fps)
    filtered = subcliped.fl(framefunc)

    return filtered,all_templates

if __name__ == "__main__":
    clippath = sys.argv[1]
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    print('loading movie')
    clip = VideoFileClip(clippath)
    print('calculating motion energy')
    m0 = motion_energy(clip)
    print(m0)
    print('filtering')
    filtered,temps = clip_filter_v2(clip,m0,m,n)
    filtered_title = clippath.split('.mp4')[0]+'filtered.mp4'
    filtered.write_videofile(filtered_title)
