import moviepy
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import sys
import re
import os
import gc
from scipy.signal import medfilt
import yaml

## Code written on 8/30. 
## Write a function object to be a single worker to which you can farm out jobs intelligently. An unfortunate consequence of moviepy processing is that VideoFileClips cannot be passed to child processes via multiprocessing, so we must pass references and load the video in each thread. If possible we should extract full clips and then throw away all full clips in each thread. 
def distribute_render(configpath,dirpath,length = 2400,threads = 4,ending= 'mpg'): 
    # First get all videos:
    files = os.listdir(dirpath)
    videos = [video for video in files if video.split('.')[-1] == ending]
    #Get configuration file for space:
    y = yaml.load(open(configpath))
    boxcoords = [y['coordinates']['box{}'.format(i)] for i in range(len(y['coordinates']))]
    
    ## Iterate through videos and collect necessary info: 
    for videopath in videos:
        ## We unfortunately have to load the clip one time in the main loop:
        clip = VideoFileClip(os.path.join(dirpath,videopath))
        ## First get the duration in seconds:
        seconds = clip.duration
        # If analysis has been found:
        all_dicts = []
        ident_base = videopath.split('.'+ending)[0]
        for ci in range(len(y['coordinates'])):
            ident =  ident_base+'roi_'+str(ci)+'cropped_'+'part'
            done = [int(re.findall('\d+',part.split('.')[0])[-1]) for part in files if ident in part.split('.')[0]]
            presegs = range(np.ceil(seconds/length).astype(int))
            if len([part for part in files if ident in part.split('.')[0]]):
                segments = [segment for segment in presegs if segment not in done]
            else:
                segments = presegs 
            ## The end segment is special:
            endseg = presegs[-1]
            tempdicts = []
            ## Now iterate through segments and recover lengths: 
            for segment in segments:
                if segment == endseg: # corner case for the last video segment. 
                    endind = -1
                else:
                    endind = length*(segment+1)
                tempdicts.append({'key':segment,'value':[segment*length,endind]})
            spatdict = {'key':ci,'value':[boxcoords[ci][ind] for ind in ['x0','x1','y0','y1']]}
            roi_dicts = [{'spatial':spatdict,'temporal':tempdict} for tempdict in tempdicts]
            all_dicts = all_dicts+ roi_dicts

        ## This returns threads different queues that parametrize jobs to be completed. 
        dicts_split = index_segments(all_dicts,threads)

        p = multiprocessing.Pool()
        ## We need to make a function object to get around the lack of lambda compatibility with multiprocessing: 
        p.map(RenderWorker(os.path.join(dirpath,videopath),os.path.join(dirpath,ident_base)),dicts_split)

class RenderWorker(object):
    def __init__(self,videopath,namebase):
        self.videopath = videopath
        self.namebase = namebase

    def __call__(self,queue):
        render_queue(queue,self.videopath,self.namebase)

def render_queue(queue,videopath,namebase):
    ## This function loads the video into memory, clips out relevant chunks, and then renders each.  
    clipqueue = []
    clip = VideoFileClip(videopath)
    basetitle = videopath.split('.')
    for i,loc in enumerate(queue):
        print('prepping' +str(i),loc)
        ## get the spatial and temporal locations referred to here. 
        ## Spatial and temporal data passed as a dict seen in the notebook you have. 
        spatfield = loc['spatial'] ## a list with four elements, denoting boundaries in pixel space. 
        spatkey = spatfield['key']
        spatval = spatfield['value']
        tempfield = loc['temporal'] ## an list with two elements, giving the start and end time boundaries in seconds.  
        tempkey = tempfield['key']
        tempval = tempfield['value']
        cropped = clip.crop(x1 = spatval[0],y1 = spatval[1],x2 = spatval[2], y2 = spatval[3])
        cropped_cutout = cropped.subclip(t_start = tempval[0],t_end = tempval[1])
        #ident =  videoname.split('.'+ending)[0]+'roi_'+str(ci)+'cropped_'+'part'
        name = namebase+'roi_'+str(spatkey)+'cropped_part'+str(tempkey)+'.mp4'
        cropped_cutout.write_videofile(name,codec = 'mpeg4',bitrate = "1500k",threads = 2)
        print('writing'+str(i),loc)

        

        

        ## queue is a set of processing chunks that the video is responsible for. It is organized as a set of tuples indicating the spatial and temporal cropping that should be handled as individual units by each thread. 



## Original version of write_cropped_video function. loads in video here. 
def write_cropped_video(cwd,video,interval,length,end): 
    clip = VideoFileClip(cwd+'/'+video)
    try:
        with open(cwd+'/'+'config.py','r+') as f:
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

        for segment in interval:
            print('moving to '+str(segment)+' of '+str(interval))
            try:
                # ensures that the last clip is the right length
                if segment == end: # corner case for the last video segment. 

                    endseg = -1
                else:
                    endseg = length*(segment+1)
                cropped_cutout = cropped.subclip(t_start = segment*length,t_end = endseg)
                cropped_cutout.write_videofile(cwd+'/'+video.split('.')[0]+'cropped_'+'part' +str(segment)+ '.mp4',codec = 'mpeg4',bitrate = "1500k",threads = 2,logger = None)

            except OSError as e:
                print('segment not viable')
            gc.collect()
    except OSError as e:

         print(e.errno)
         print('configuration not loaded')

def write_cropped_video_new(cliplist,basetitle,segments):
    for si,segment in enumerate(segments):
        print(basetitle+str(segment)+'.mp4')
        cliplist[si].write_videofile(basetitle+str(segment)+'.mp4',codec = 'mpeg4',bitrate = "1500k",threads = 4)
        #cropped_cutout.write_videofile(cwd+'/'+video.split('.')[0]+'cropped_'+'part' +str(segment)+ '.mp4',codec = 'mpeg4',bitrate = "1500k",threads = 2,logger = None)
# No lambda functions are allowed in parallelized code, so we will use a function object instead. 
class Renderer(object):
    def __init__(self,cwd,video,length,end):
        self.cwd = cwd 
        self.video = video
        self.length = length
        self.end = end
    def __call__(self,segments):
        write_cropped_video(self.cwd,self.video,segments,self.length,self.end)

# New renderer class. Takes in a list of clips (is this prohibitive?) and a base title. Call method takes the segment indices as a generator.) 
class Renderer_new(object):
    def __init__(self,cliplist,basetitle):
        self.cliplist = cliplist 
        self.basetitle = basetitle
    def __call__(self,segments):
        write_cropped_video_new(self.cliplist,self.basetitle,self.segments)

# helper function for parallelization: 
def index_segments(segments,vcpus):
    nb_partition = np.max((np.ceil(len(segments)/vcpus).astype(int),1))
    list_parts  = [segments[i*nb_partition:(i+1)*nb_partition] for i in range((len(segments)+nb_partition-1)//nb_partition)]
    return list_parts

## Crop a certain portion of the video
def crop_video(configpath,clip):
    y = yaml.load(open(configpath))
    ## Get coordinates: 
    coords = y['coordinates']
    basebox = 'box{}'
    clips = []
    for i in range(len(coords)):
        boxname = basebox.format(str(i))
        boxcoords = coords[boxname]
        cropped = clip.crop(x1 = boxcoords['x0'],x2 = boxcoords['x1'],y1 = boxcoords['y0'],y2 = boxcoords['y1'])
        clips.append(cropped)
    return clips

## Write a function that chops up a clip into parts indexed by the segment and the end segment index. 
def cut_clip_segments(cropped,segments,endseg,length):
    all_clips = []
    for segment in segments:
        print('moving to '+str(segment))
        try:
            # ensures that the last clip is the right length
            if segment == endseg: # corner case for the last video segment. 
                endind = -1
            else:
                endind = length*(segment+1)
            cropped_cutout = cropped.subclip(t_start = segment*length,t_end = endind)
            all_clips.append(cropped_cutout)
        except OSError as e:
            print('segment not viable')
        gc.collect()
    return all_clips

## Write a wrapper function that takes a bunch of clips, a set of indices, a base title, and writes the clip.  

## Newest method to cut videos as of 8/29.
## Assumes that there is more than one ROI in the frame, and iterates over each roi as an independent clip. 
def cut_videos_new(configpath,dirpath,length = 1200,threads = 4,ending= 'mpg'):
    """
    configpath: the path to the config file.  
    dirpath: the path to the directory we want to use. 
    length: the length of the resulting clips in seconds. 
    threads: the number of threads to use for processing. 
    """
    # First get all videos:
    files = os.listdir(dirpath)
    videos = [video for video in files if video.split('.')[-1] == ending]
    for videoname in videos:
        ## Load in the video: 
        clip = VideoFileClip(os.path.join(dirpath,videoname))
        ## Now crop it according to the configpath: 
        croppedlist = crop_video(configpath,clip)
        print(videoname,croppedlist)
        ## Now for each of the cropped videos, we will look at the motion energy in each: 
        for ci,cropclip in enumerate(croppedlist):
            ## First get the duration in seconds:
            seconds = clip.duration
            # If analysis has been found:
            ident =  videoname.split('.'+ending)[0]+'roi_'+str(ci)+'cropped_'+'part'
            done = [int(re.findall('\d+',part.split('.')[0])[-1]) for part in files if ident in part.split('.')[0]]
            presegs = range(np.ceil(seconds/length).astype(int))
            if len([part for part in files if ident in part.split('.')[0]]):
                segments = [segment for segment in presegs if segment not in done]
            else:
                segments = presegs 
            ## explicitly save the last segment index. 
            endseg = presegs[-1]
            ## Iterate through segment indices and cut up the video accordingly: 
            clipsegs = cut_clip_segments(cropclip,segments,endseg,length)
             
            # This determines how we will split our resources: 
            segments_split = index_segments(segments,threads)

            print(segments)
            #write_cropped_video_new(clipsegs,os.path.join(dirpath,ident),segments)
            print('Parallelizing video processing into '+str(len(segments_split))+' different threads')
            p = multiprocessing.Pool()
            ### We need to make a function object to get around the lack of lambda compatibility with multiprocessing: 
            p.map(Renderer_new(clipsegs,os.path.join(dirpath,ident)),segments_split)
            #print('Done')

            ## Now we will distribute computation over the 4 virtual threads and 
            # now generate splits:
            #print(list(segments_split[0]))
            # Chunk the video: 

            ## Now we will write cropped files:  
            print(segments_split)
            #print('Parallelizing video processing into '+str(len(segments_split))+' different threads')
            #p = multiprocessing.Pool()


        

    


def cut_videos_p(configpath,cwd = None):
    # length in seconds of each video 
    length = 1200
    threads = 4
    if cwd is None:
        # First get current directory
        cwd = os.getcwd()
    # First get all subdirectories:
    files = os.listdir(cwd)
    videos = [video for video in files if video.split('.')[-1] == 'avi']
    print(videos)
    for video in videos:
        ## Load in video
        print('loading ' +video)
        clip = VideoFileClip(cwd+'/'+video)
        try: 
            with open(configpath,'r+') as f:
                coords = ['x0 = \n','y0 = \n','x1 = \n','y1 = \n']
                intcoords = []
                for coord in range(len(coords)):
                    coords[coord] = f.readline()
                    nums = re.findall('\d+',coords[coord])[1]
                    intcoords.append(nums)
            

        except OSError as e: 
            print(e.errno)
            print('configuration not loaded')
        
        ## First get the duration in seconds:
        seconds = clip.duration
        # If analysis has been found:
        ident =  video.split('.')[0]+'cropped_'+'part'
        print(files,ident)
        print([part for part in files if ident in part.split('.')[0]])
        done = [int(re.findall('\d+',part.split('.')[0])[-1]) for part in files if ident in part.split('.')[0]]
        presegs = range(np.ceil(seconds/length).astype(int))
        if len([part for part in files if ident in part.split('.')[0]]):
            segments = [segment for segment in presegs if segment not in done]
        else:
            segments = presegs 
        # This determines how we will split our resources: 
        segments_split = index_segments(segments,threads)
        print('Parallelizing video processing into '+str(len(segments_split))+' different threads')
        p = multiprocessing.Pool()
        ## We need to make a function object to get around the lack of lambda compatibility with multiprocessing: 
        p.map(Renderer(cwd,video,length,presegs[-1]),segments_split)
        print('Done')


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
