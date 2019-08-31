'''
Object class to evaluate processed data against the training data. This evaluation takes in a set of social_dataset objects and a training data object, and can evaluate the training data object against the social_dataset object, considering body model violations, speed violations, and image model violations. 
'''
import numpy as np 
import sys
sys.path.insert(0,'../')
import moviepy
from tqdm import tqdm
from Social_Dataset_Class_v2 import find_segments

# KL divergence function for image histograms. 
def empirical_KL(groundtruth,empirical):
    ## We assume that the words are spit out directly from the numpy.histogram function.
    # First check that the x axis is the same: 
    assert np.all(groundtruth[0][1]==empirical[0][1]), 'The x axis must be the same for both sets of distributions'
    # Iterate over all three color channels
    kl_tot = 0
    for i in range(3):
        # The quantity is only defined over the support of the groundtruth distribution
        support = np.where(groundtruth[i][0])
        g_normed = groundtruth[i][0]/np.sum(groundtruth[i][0]).astype('float')
        e_normed = empirical[i][0]/np.sum(empirical[i][0]).astype('float')
        g_vals = g_normed[support]
        e_vals = e_normed[support]
    
        assert len(g_vals) == len(e_vals)
        logratio = np.log((e_vals+1e-8)/g_vals)
        kl = -np.dot(g_vals,logratio)
        kl_tot += kl
    return kl_tot

# TODO does not see intervals that are omitted from the end. 
def empty_intervals(segments):
    diffs = []
    last = 0
    for segment in segments:
        diff = segment[0]-last
        last = segment[-1]
        diffs.append(diff)
    return diffs

class Evaluation(object):
    def __init__(self,training_data,social_datasets):
        self.training_data = training_data
        self.social_datasets = social_datasets ## We can evaluate multiple datasets at once. 
        self.xmin = 0
        self.xmax = 0
        self.ymin = 0
        self.ymax = 0

    ### We will divide evaluation into three separate categories: body model evaluation, speed evaluation, and image evaluation. 
    ## Body model evaluation. Returns a dictionary containing part pairs as keys and deviance over time as values.

    def set_cropping(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def evaluate_bodymodel(self):   
        # Pull the actual part distances:
        dist_dict = {}
        for mouse in range(2):
            for j in range(5):
                for i in range(j):
                    value = []
                    part0 = j+mouse*5
                    part1 = i+mouse*5
                    for dataset in range(len(self.social_datasets)):
                        distance = self.social_datasets[dataset].part_dist(part0,part1)
                        key = (self.social_datasets[dataset].part_list[part0],self.social_datasets[dataset].part_list[part1])
                        value.append(distance)
                    value = np.concatenate(value)
                    dist_dict[key] = value
        # Calculate the error bars on this: 
        stats_v,stats_m = [self.training_data.stats_wholemouse([i+1 for i in range(100)],m) for m in range(2)]
                    
        return dist_dict,stats_v,stats_m

    ## Speed evaluation. I don't really know how to incorporate the training data here, so let's just focus on ways to produce the distribution. 
    def evaluate_speed(self):
        speed_dict = {}
        # Pull the differences in the velocity:
        for mouse in range(2):
            for j in range(5):
                speed = []
                part = j+mouse*5
                for dataset in range(len(self.social_datasets)):
                    trajectory = self.social_datasets[dataset].render_trajectory_full(part)
                    velocity = np.diff(trajectory,axis=0)
                    speed.append(np.linalg.norm(velocity,axis = 1))
                speed = np.concatenate(speed)
                speed_dict[self.social_datasets[dataset].part_list[part]] = speed
        return speed_dict
    
    ## Interval evaluation. Once again, this is sort of separate from the training data, but regardless...  
    def evaluate_segments(self):
        interval_dict = {}
        ## Pull intervals from the trajectory:
        for dataset in range(len(self.social_datasets)):
            good_indices = self.social_datasets[dataset].allowed_index_full
            for mouse in range(2):
                for j in range(5):
                    part = j+mouse*5
                    indices = good_indices[part][:,0]
                    intervals = find_segments(indices)
                    empty_space = empty_intervals(intervals)
                    interval_dict[self.social_datasets[dataset].part_list[part]] = empty_space
        return interval_dict

    ## Now, we evaluate according to the distribution of pixels underlying the tracked groundtruth points 

    def extract_patches(self,dataset_index,part_index,radius):
        dataset = self.social_datasets[dataset_index]
        trajectory = dataset.render_trajectory_full(part_index) 
        all_clipped = np.zeros((int(dataset.movie.fps*dataset.movie.duration),2*radius,2*radius,3)).astype(np.uint8)
        for framenb,position in tqdm(enumerate(trajectory[:1000])):
            frame = dataset.movie.get_frame(framenb/float(dataset.movie.fps))
            ysize,xsize = frame.shape[:2]
            xcent,ycent = position[0]-self.xmin,position[1]-self.ymin
            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## Edge detection:
            pads = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])

            clip = frame[ymin:ymax,xmin:xmax]

            if np.any(pads < 0):
                topad = pads < 0
                padding = -1*pads*topad
                if np.any(np.array(np.shape(clip)) == 0):
                    
                    clip = np.zeros((2*radius,2*radius,3))
                else:
                    clip = np.pad(clip,padding,'edge')

            # Compile:
            all_clipped[framenb,:,:,:] = (np.round(255*clip)).astype(np.uint8)
        return all_clipped

    def patch_grandhist(self,dataset_index,part_index,radius):
        patches = self.extract_patches(dataset_index,part_index,radius)
        hists = [np.histogram(patches[:,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)]
        return hists


    def patch_hist(self,dataset_index,part_index,radius):
        patches = self.extract_patches(dataset_index,part_index,radius)
        hists = [[np.histogram(patches[frame,:,:,i],bins = np.linspace(0,255,256)) for i in range(3)] for frame in range(len(patches))]
        return hists

    def evaluate_image(self):
        ## For each body part, we first extract the groundtruth histograms:
        image_dict = {}
        for mouse in range(3):
            for i in range(5):
                part = mouse*5+i
                radius = 4
                groundtruth_hist = self.training_data.patch_grandhist(range(100),part,radius)
                kl_total = []

                for dataset in range(len(self.social_datasets)):
                    # Now extract histograms from the video:
                    emp_hist = self.patch_hist(dataset,part,radius)
                    kls = np.zeros(len(emp_hist))
                    for nb_hist,hist in tqdm(enumerate(emp_hist)):
                        kl = empirical_KL(groundtruth_hist,hist)
                        kls[nb_hist] = kl
                    kl_total.append(kls)
                image_dict[self.social_datasets[dataset].part_list[part]] = kl_total
        return image_dict



import numpy 
numpy.random.randn(10)
