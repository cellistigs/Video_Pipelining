import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import os
import moviepy
import glob
from moviepy.editor import VideoClip,VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage
from PIL import Image
import matplotlib.gridspec as gridspec
from skimage.util import img_as_ubyte
from skimage.transform import rescale
from scipy.ndimage import rotate
import tensorflow as tf
import sys


### Geometry helper functions
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def unit_vector_vec(vectorvec):
    return vectorvec / np.linalg.norm(vectorvec,axis = 1,keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_vec(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector_vec(v1)
    v2_u = unit_vector_vec(v2)
    dotted = np.einsum('ik,ik->i', v1_u, v2_u)
    return np.arccos(np.clip(dotted, -1.0, 1.0))

## Tensorflow Data API helper functions:
## Designate helper function to define features for examples more easily
def _int64_feature_(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature_(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature_(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))



def _write_ex_animal_focused(videoname,i,image,mouse,pos):
    features = {'video': _bytes_feature_(tf.compat.as_bytes((videoname))),
                'frame': _int64_feature_(i),
                'image': _bytes_feature_(tf.compat.as_bytes(image.tobytes())),
                'mouse': _int64_feature_(mouse),
                'position':_bytes_feature_(tf.compat.as_bytes(pos.tobytes()))
                }
    example = tf.train.Example(features = tf.train.Features(feature = features))
    return example

## indexing helper functions:
def sorter(index,bp = 5,cent_ind = 3):
    # return the center index and other body part indices given a mouse index
    center = index*bp+3
    all_body = np.arange(bp)+index*bp
    rel = list(all_body)
    rel.pop(cent_ind)
    return center,rel

class social_dataset(object):
    def __init__(self,filepath,vers = 0):
        self.dataset = pd.read_hdf(filepath)
        self.dataset_name = filepath.split('/')[-1]
        self.scorer = 'DeepCut' + filepath.split('DeepCut')[-1].split('.h5')[0]
        self.part_list = ['vtip','vlear','vrear','vcent','vtail','mtip','mlear','mrear','mcent','mtail']
        self.part_index = np.arange(len(self.part_list))
        self.part_dict = {index:self.part_list[index] for index in range(len(self.part_list))}
        self.time_index = np.arange(self.dataset.shape[0])
        self.allowed_index = [self.time_index[:,np.newaxis] for _ in self.part_list] ## set of good indices for each part!
        self.allowed_index_full = [self.simple_index_maker(part,self.allowed_index[part]) for part in self.part_index]
        self.vers = vers ## with multiple animal index or not

    # Helper function: index maker. Takes naive index constructs and returns a workable set of indices for the dataset.
    def simple_index_maker(self,pindex,allowed):
        length_allowed = len(allowed)
        # First compute the part index as appropriate:
        part = np.array([[pindex]]).repeat(length_allowed,axis = 0)

        full_index = np.concatenate((allowed,part),axis = 1)
        return full_index


    # Trajectory selector:
    def select_trajectory(self,pindex):
        part_name = self.part_dict[pindex]
        part_okindex = self.allowed_index[pindex]
        if self.vers == 0:
            rawtrajectory = self.dataset[self.scorer][part_name]['0'].values[:,:2]
        else:
            rawtrajectory = self.dataset[self.scorer][part_name].values[:,:2]
        return rawtrajectory

    # Define a trajectory rendering function for any part:

#     def render_trajectory(self,pindex):
#         rawtrajectory = self.select_trajectory(pindex)
#         part_name = self.part_dict[pindex]
#         part_okindex = self.allowed_index[pindex]

#         allowed_data = rawtrajectory[part_okindex,:]
#         filtered_x,filtered_y = np.interp(self.time_index,part_okindex,allowed_data[:,0]),np.interp(self.time_index,part_okindex,allowed_data[:,1])
#         filtered_part = np.concatenate((filtered_x[:,np.newaxis],filtered_y[:,np.newaxis]),axis = 1)
#         return filtered_part

    # if a trajectory could be influenced by other trajectories:
    def render_trajectory_full(self,pindex):
        rawtrajectories = self.dataset[self.scorer].values
        part_okindex = self.allowed_index_full[pindex]
        time = part_okindex[:,0:1]
        x = part_okindex[:,1:2]*3
        y = part_okindex[:,1:2]*3+1
        coords = np.concatenate((x,y),axis = 1)
        out = rawtrajectories[time,coords]
        filtered_x,filtered_y = np.interp(self.time_index,part_okindex[:,0],out[:,0]),np.interp(self.time_index,part_okindex[:,0],out[:,1])
        filtered_part = np.concatenate((filtered_x[:,np.newaxis],filtered_y[:,np.newaxis]),axis = 1)
        return filtered_part

    # For multiple trajectories:

    def render_trajectories(self,to_render = None):
        if to_render == None:
            to_render = self.part_index
        part_trajectories = []
        for pindex in to_render:
            part_traj = self.render_trajectory_full(pindex)
            part_trajectories.append(part_traj)

        return(part_trajectories)


    # Now define a plotting function:
    # Also plot a tracked frame on an image:
    def plot_image(self,part_numbers,frame_nb):
        allowed_indices = [frame_nb in self.allowed_index_full[part_number][:,0] for part_number in part_numbers]
        colors = ['red','blue']
        point_colors = [colors[allowed] for allowed in allowed_indices]
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        print(relevant_trajectories[0].shape)
        relevant_points = [traj[frame_nb,:] for traj in relevant_trajectories]
        relevant_points = np.array(relevant_points)
        assert np.all(np.shape(relevant_points) == (len(part_numbers),2))
        print(self.movie.duration,self.movie.fps)

        # Now load in video:
        try:
            frame = self.movie.get_frame(frame_nb/self.movie.fps)
            fig,ax = plt.subplots()
            ax.imshow(frame)
            ax.axis('off')
            ax.scatter(relevant_points[:,0],relevant_points[:,1],c = point_colors)
            plt.show()
        except OSError as error:
            print(error)

    # Plot a tracked frame on an image, with raw trackings for comparison:
    def plot_image_compare(self,part_numbers,frame_nb):
        allowed_indices = [frame_nb in self.allowed_index_full[part_number][:,0] for part_number in part_numbers]
        colors = ['red','blue']
        shapes = ['o','v']
        point_colors = [colors[allowed] for allowed in allowed_indices]
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        relevant_points = [traj[frame_nb,:] for traj in relevant_trajectories]
        relevant_points = np.array(relevant_points)

        rawtrajectories = self.dataset[self.scorer].values[frame_nb,:]
        a = rawtrajectories.reshape(10,3)
        relevant_raw = np.array([part[:2] for i, part in enumerate(a) if i in part_numbers])
        print(relevant_raw.shape)
        assert np.all(np.shape(relevant_points) == (len(part_numbers),2))
        assert np.all(np.shape(relevant_raw) == (len(part_numbers),2))
        print(self.movie.duration,self.movie.fps)

        # Now load in video:
        try:
            frame = self.movie.get_frame(frame_nb/self.movie.fps)
            fig,ax = plt.subplots()
            ax.imshow(frame)
            ax.axis('off')
            ax.scatter(relevant_points[:,0],relevant_points[:,1],c = point_colors)
            ax.scatter(relevant_raw[:,0],relevant_raw[:,1],c = point_colors,marker = 'v')
            plt.show()
        except OSError as error:
            print(error)

    def plot_trajectory(self,part_numbers,start = 0,end = -1,cropx = 0,cropy = 0,axes = True,save = False,**kwargs):
        # First define the relevant part indices:
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        names = ['Virgin','Mother']
        mouse_id = part_numbers[0]/5
        if axes == True:
            fig,axes = plt.subplots()
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:end,0],-trajectory[start:end,1],label = self.part_list[part_numbers[part_nb]],**kwargs)
            axes.axis('equal')
            plt.legend()
            plt.title(names[mouse_id]+' Trajectories')
#             plt.yticks(0,[])
#             plt.xticks(0,[])
            plt.show()
            plt.close()
            if save != False:
                plt.savefig(save)

        else:
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:end,0]-cropx,trajectory[start:end,1]-cropy,label = self.part_list[part_numbers[part_nb]],**kwargs)

## Closely related to the plotting function is the gif rendering function for trajectories:
    def gif_trajectory(self,part_numbers,start = 0,end = -1,fps = 60.,cropx = 0,cropy = 0,save = False,**kwargs):
        # First define the relevant part indices:
        relevant_trajectories = self.render_trajectories(to_render = part_numbers)
        names = ['Virgin','Mother']
        mouse_id = part_numbers[0]/5
        if end == -1:
            duration = relevant_trajectories.shape[0]-start
        else:
            duration = end - start

        clipduration = duration/fps

        fig,axes = plt.subplots()
        print(relevant_trajectories[0][start:start+2+int(5*fps),0].shape)
        ## Define a frame making function to pass to moviepy:
        def gif_trajectory_mini(t):
            axes.clear()
            for part_nb,trajectory in enumerate(relevant_trajectories):
                axes.plot(trajectory[start:start+2+int(t*fps),0],-trajectory[start:start+2+int(t*fps),1],label = self.part_list[part_numbers[part_nb]],**kwargs)
                axes.plot(trajectory[start+int(t*fps),0],-trajectory[start+int(t*fps),1],'o',markersize = 3,label = self.part_list[part_numbers[part_nb]],**kwargs)
            axes.axis('equal')

            return mplfig_to_npimage(fig)

        animation = VideoClip(gif_trajectory_mini,duration = clipduration)
        # animation.ipython_display(fps=60., loop=True, autoplay=True)
        return animation




    def calculate_speed(self,pindex):
        rawtrajectory = self.select_trajectory(pindex)
        diff = np.diff(rawtrajectory,axis = 0)
        speed = np.linalg.norm(diff,axis = 1)
        return speed

    def motion_detector(self,threshold = 80):
        ## We will assume the trajectories are raw.
        indices = np.array([0,1,2,3,4])
        mouse_moving = np.zeros((2,))
        for mouse in range(2):
            mouseindices = indices+5*mouse
            trajectories = np.concatenate(self.render_trajectories(list(indices)),axis = 1)

            maxes = np.max(trajectories,axis = 0)
            mins = np.min(trajectories,axis = 0)
            differences = maxes-mins
            partwise_diffs = differences.reshape(5,2)
            normed = np.linalg.norm(partwise_diffs,axis = 1)
            ## See if any of the body parts left
            if np.any(normed <threshold):
                mouse_moving[mouse] = 0
            else:
                mouse_moving[mouse] = 1

        return mouse_moving,normed

    # Now we define filtering functions:
    # Filter by average speed
    def filter_speed(self,pindex,threshold = 10):
        rawtrajectory = self.select_trajectory(pindex)
        diff = np.diff(rawtrajectory,axis = 0)
        speed = np.linalg.norm(diff,axis = 1)
        filterlength = 9

        filterw = np.array([1.0 for i in range(filterlength)])/filterlength
        filtered = np.convolve(speed,filterw,'valid')
        # virg_outs = np.where(filtered > 10)[0]
        outliers = np.where(filtered>threshold)[0]

        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outliers])

        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    # Filter by likelihood
    def filter_likelihood(self,pindex):
        part_name = self.part_dict[pindex]
        part_okindex = self.allowed_index_full[pindex]
        if self.vers == 0:
            likelihood = self.dataset[self.scorer][part_name]['0'].values[:,2]
        else:
            likelihood = self.dataset[self.scorer][part_name].values[:,2]
        outliers = np.where(likelihood<0.95)[0]
        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outliers])

        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    def filter_nests(self):
        try:
            print("nest bounds are: "+str(self.bounds))
            indices = np.array([0,1,2,3,4])
            for mouse in range(2):
                mouseindices = indices+5*mouse
                ## Bounds are defined as x greater than some value, y greater than some value (flipped on plot)
                self.filter_speeds(mouseindices,9)
                whole_traj = self.render_trajectories(list(mouseindices))
                for part_nb,traj in enumerate(whole_traj):
                    pindex = part_nb+mouse*5
                    ## Discover places where the animal is in its nest
                    bounds_check = traj-self.bounds
                    checkarray = bounds_check>0
                    in_nest = checkarray[:,0:1]*checkarray[:,1:]
                    nest_array = np.where(in_nest)[0]
                    okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in nest_array])
                    self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

        except NameError as error:
            print(error)


    def filter_speeds(self,indices,threshold = 10):
        print('filtering by speed...')
        for index in indices:
            self.filter_speed(index,threshold)
    def filter_likelihoods(self,indices):
        print('filtering by likelihood...')
        for index in indices:
            self.filter_likelihood(index)

    def reset_filters(self,indices):
        print('resetting filters...')
        for index in indices:
            self.allowed_index_full[index] = self.simple_index_maker(index,self.time_index[:,np.newaxis])


    def interhead_position(self,mouse_id):
        # Returns positions between the head for a given mouse
        part_id = 5*mouse_id
        lear,rear = self.render_trajectories([part_id+1,part_id+2])

        stacked = np.stack((lear,rear))
        centroids = np.mean(stacked,axis = 0)
        return centroids
    # Vector from the center of the head to the tip.
    def head_vector(self,mouse_id):
        # First get centroid:
        head_cent = self.interhead_position(mouse_id)

        vector = self.render_trajectory_full(mouse_id*5)-head_cent
        return vector

    ## Vector from the center of the body to the center of the head.
    def body_vector(self,mouse_id):
        vector = self.render_trajectory_full(mouse_id*5)-self.render_trajectory_full(mouse_id*5+3)
        return vector

    def head_angle(self,mouse_id):
        # First get centroid:
        vector = self.head_vector(mouse_id)
        north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
        angles = angle_between_vec(north,vector)
        return angles

    def body_angle(self,mouse_id):
        vector = self.body_vector(mouse_id)
        north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
        sign = np.sign(north[:,1]-vector[:,1])
        angles = angle_between_vec(north,vector)*sign
        return angles

    # Gives the relative position of the other mouse with regard to the mouse provided in mouse_id
    def relative_vector(self,mouse_id):
        #First get centroid:
        head_cent = self.interhead_position(mouse_id)
        # Get other mouse centroid
        other_mouse = abs(mouse_id-1)
        other_mouse_centroid= self.render_trajectory_full(5*other_mouse)
        vector = other_mouse_centroid-head_cent
        return vector

    def gaze_relative(self,mouse_id):
        head_vector = self.head_vector(mouse_id)
        rel_vector = self.relative_vector(mouse_id)
        angles = angle_between_vec(head_vector,rel_vector)

        return angles
    # Import the relevant movie file
    def import_movie(self,moviepath):
        self.movie = VideoFileClip(moviepath)
#         self.movie.reader.initialize()
    # Plot the movie file, and overlay the tracked points
    def show_frame(self,frame,plotting= True):
        image = img_as_ubyte(self.movie.get_frame(frame/self.movie.fps))
        fig,ax = plt.subplots()
        ax.imshow(image[70:470,300:600])
#         ax.imshow(image)
        if plotting == True:
            self.plot_trajectory([0,1,2,3,4,5,6,7,8,9],start = frame,end = frame+1,cropx = 300,cropy = 70,axes = ax,marker = 'o')
        plt.show()

    def deviance_final(self,i,windowlength,reference,ref_pindex,target,target_pindex):
        ## We have to define all relevant index sets first. This is actually where most of the trickiness happens. We do the
        ## following: 1) define a starting point in the TARGET trajectory, by giving an index into the set of allowed axes.
        # First define the relevant indices for interpolation: return the i+1th and the windowlength+i+1th index in the test set:
        sample_indices_absolute = [i+1,windowlength+i+1]
        ref_indices = ref_pindex[:,0]

        target_indices = target_pindex[:,0]

        test_indices_sample_start = target_indices[sample_indices_absolute[0]]
        test_indices_sample_end = target_indices[sample_indices_absolute[-1]]

        ## We have to find the appropriate indices in the reference trajectory: those equal to or just outside the test indices
        start_rel_ref,end_rel_ref = np.where(ref_indices <= test_indices_sample_start)[0][-1],np.where(ref_indices >= test_indices_sample_end)[0][0]

        sample_indices_rel = ref_indices[[start_rel_ref-1,start_rel_ref,end_rel_ref-1,end_rel_ref]]

        ## Define the relevant indices for comparison in the test trajectory space:
        comp_indices_rel_test = target_indices[sample_indices_absolute[0]:sample_indices_absolute[-1]]



        ## Now we should use indices to 1) interpolate the baseline trajectory, 2) evaluate the fit of the test to the
        ## interpolation
        traj_ref = reference
        traj_test = target

        traj_ref_sample = traj_ref[[start_rel_ref-1,start_rel_ref,end_rel_ref,end_rel_ref+1],:]

        ## Create interpolation function:
        f = interp1d(sample_indices_rel,traj_ref_sample,axis = 0)

        ## Now evaluate this at the relevant points on the test function!


        interped_points = f(comp_indices_rel_test)

        sampled_points = traj_test[sample_indices_absolute[0]:sample_indices_absolute[-1],:]
        return interped_points,sampled_points,comp_indices_rel_test

    def filter_check(self,pindex,windowlength,varthresh):

        sample_indices = lambda i: [i,i+1,windowlength+i+1,windowlength+i+2]
        # Define the indices you want to check: acceptable indices in both the part trajectory for this animal and
        # its reference counterpoint
        mouse_nb = pindex/5

        all_vars = []
        all_outs = []


        target = self.render_trajectory_full(pindex)
        target_indices = self.allowed_index_full[pindex]

        compare_max = len(target_indices)-windowlength-2
        for i in range(compare_max):

            current_vars = []
            current_outs = []
            for j in range(2):
                interped,sampled,indices = self.deviance_final(i,windowlength,target,target_indices,target,target_indices)
                linewise_var = np.max(np.max(abs(interped-sampled),axis = 0))
                current_vars.append(linewise_var)
                if linewise_var > varthresh:
                    mis = np.where(abs(interped-sampled)>varthresh)[0]
                    if not list(mis):
                        pass
                    else:
                        current_outs.append(indices[mis])
                if not current_outs:
                    pass
                else:
                    all_outs.append(current_outs)
        return all_outs
    # Cross check against the centroid of the other mouse:
    def filter_crosscheck(self,pindex):
        windowlength = 15
        sample_indices = lambda i: [i,i+1,windowlength+i+1,windowlength+i+2]
        # Define the indices you want to check: acceptable indices in both the part trajectory for this animal and
        # its reference counterpoint
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        mice = [pindex,other_pindex]
        all_vars = []
        all_outs = []

        reference_centroid = self.render_trajectory_full(other_pindex)
#         target_centroid = self.render_trajectory_full(mouse_nb*5+3)
        target = self.render_trajectory_full(pindex)

        reference_centroid_indices = self.allowed_index_full[other_pindex]
#         target_centroid_indices = self.allowed_index_full[mouse_nb*5+3]
        target_indices = self.allowed_index_full[pindex]

#         numerical_allowed_ref = [index for index in self.allowed_index_full[pindex][:,0] if index+14<len(reference)] ## we have to skim at the end
        numerical_allowed_tar = [index for index in self.allowed_index_full[other_pindex][:,0] if index+14<len(target)] ## we have to skim at the end
        ## Ugly: we need to make sure that we don't index past what we can compare to....

#         compare_max = np.min([len(target_indices)-windowlength-1,len(reference_indices)-windowlength-1])
        compare_max = np.min([len(target_indices)-windowlength-2,len(reference_centroid_indices)-windowlength-2])

        for i in range(compare_max):

            current_vars = []
            current_outs = []
            for j in range(2):
                interped,sampled,indices = self.deviance_final(i,windowlength,target,target_indices,target,target_indices)
                linewise_var = np.max(np.max(abs(interped-sampled),axis = 0))
                current_vars.append(linewise_var)
                if linewise_var > 20:
#                     mis = np.where(abs(interped-sampled)>20)[0]
                    # Cross check with the other trajectory:
                    cross_interped,cross_sampled,indices = self.deviance_final(i,windowlength,reference_centroid,reference_centroid_indices,target,target_indices)
                    # Now check where the points are closer to this cross trajectory than the current one:
                    std_diff = np.linalg.norm(interped-sampled,axis = 1)
                    cross_diff = np.linalg.norm(cross_interped-cross_sampled,axis = 1)
                    # Look for points where the point is __significantly__ closer to the cross than the standard linearization:
                    ratio = cross_diff/std_diff
                    mis = np.where(ratio<0.5)[0]
                    if not list(mis):
                        pass
                    else:
                        current_outs.append(indices[mis])
                if not current_outs:
                    pass
                else:
                    all_outs.append(current_outs)
        return all_outs

    def filter_crosscheck_replace(self,pindex):
        mouse_nb = pindex/5
        other_mouse = abs(mouse_nb-1)
        other_pindex = int((other_mouse-mouse_nb)*5+pindex)
        outs = self.filter_crosscheck(pindex)
        outs_processed = ([element for out in outs for element in out])
        if not outs_processed:
            pass
        else:
            outs_processed = np.unique(np.concatenate(outs_processed))

        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outs_processed])
        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])
#         invouts = self.filter_crosscheck(other_pindex))
#         print(out.intersection(invouts))
#         print(set(outs).intersection(set(invouts)),"printing")
    def filter_crosscheck_replaces(self,indices):
        for pindex in indices:
            print('crosschecking '+ str(self.part_dict[pindex]))
            self.filter_crosscheck_replace(pindex)

    def filter_check_replace(self,pindex,windowlength= 30,varthresh=30):
        mouse_nb = pindex/5
        outs = self.filter_check(pindex,windowlength,varthresh)
        outs_processed = ([element for out in outs for element in out])
        if not outs_processed:
            pass
        else:
            outs_processed = np.unique(np.concatenate(outs_processed))
        print(outs_processed)
        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outs_processed])
        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    def filter_check_replaces(self,indices):
        for pindex in indices:
            print('checking '+ str(self.part_dict[pindex]))
            self.filter_check_replace(pindex)

    def filter_check_v2(self,pindex,windowlength,varthresh):

        sample_indices = lambda i: [i,i+1,windowlength+i+1,windowlength+i+2]
        # Define the indices you want to check: acceptable indices in both the part trajectory for this animal and
        # its reference counterpoint
        mouse_nb = pindex/5

        all_vars = []
        all_outs = []


        target = self.render_trajectory_full(pindex)
        target_indices = self.allowed_index_full[pindex]

        compare_max = len(target_indices)-windowlength-2
        for i in range(compare_max):

            current_vars = []
            current_outs = []
            for j in range(2):
                interped,sampled,indices = self.deviance_final(i,windowlength,target,target_indices,target,target_indices)
                linewise_var = np.max(np.max(abs(interped-sampled),axis = 0))
                current_vars.append(linewise_var)
                if linewise_var > varthresh:
                    mis = -1*(np.max((abs(interped-sampled)>varthresh),axis = 1)*2-1)
                else:
                    mis = np.ones(np.shape(interped)[0])
                scores[i+1:i+windowlength+1] += mis
        return scores

    def filter_check_replace_v2(self,pindex,windowlength= 45,varthresh=40):
        mouse_nb = pindex/5
        preouts = self.filter_check(pindex,windowlength,varthresh)
        outs = np.where(preouts<0)[0]
        if not len(outs):
            pass
        else:
            outs = np.unique(outs)

        okay_indices = np.array([index for index in self.allowed_index_full[pindex] if index[0] not in outs])
        self.allowed_index_full[pindex] = self.simple_index_maker(pindex,okay_indices[:,0:1])

    def filter_check_replaces_v2(self,indices,windowlength= 45,varthresh=40):
        for pindex in indices:
            print('checking '+ str(self.part_dict[pindex]))
            self.filter_check_replace_v2(pindex,windowlength= 45,varthresh=40)

    def return_cropped_view(self,mice,frame,radius = 64):
        mouse_views = []
        image = img_as_ubyte(self.movie.get_frame((frame)/self.movie.fps))
        for mouse in mice:
            ## Image Plots:
            all_cents = self.render_trajectory_full(mouse*5+3)
            xcent,ycent = all_cents[frame,0],all_cents[frame,1]
    #         if (xcent < 550) and (ycent<400):

            xsize,ysize = self.movie.size
            xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]
            if np.any(pads < 0):
        #         print('ehre')
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')
            mouse_views.append(clip)
        return mouse_views

    def return_cropped_view_rot(self,mice,frame,angle,radius = 64):
        mouse_views = []
        image = img_as_ubyte(self.movie.get_frame((frame)/self.movie.fps))
        for mouse in mice:
            ## Image Plots:
            all_cents = self.render_trajectory_full(mouse*5+3)
            xcent,ycent = all_cents[frame,0],all_cents[frame,1]
    #         if (xcent < 550) and (ycent<400):

            xsize,ysize = self.movie.size
            buffer = np.ceil(np.sqrt(2))
            xmin,xmax,ymin,ymax = int(xcent-buffer*radius),int(xcent+buffer*radius),int(ycent-buffer*radius),int(ycent+buffer*radius)
            ## do edge detection:
            pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])


            clip = image[ymin:ymax,xmin:xmax]
            if np.any(pads < 0):
        #         print('ehre')
                topad = pads<0
                padding = -1*pads*topad
                clip = np.pad(clip,padding,'edge')

            clip_rot = rotate(clip,angle,reshape = False)
            print(clip_rot.shape)
            xminf,xmaxf,yminf,ymaxf = int(buffer*radius-radius),int(buffer*radius+radius),int(buffer*radius-radius),int(buffer*radius+radius)
            clip_final = clip_rot[yminf:ymaxf,xminf:xmaxf]
            mouse_views.append(clip_final)
        return mouse_views

    def write_cropped_rotated_tfrecords(self,mouse,filename,cut =None):
        writer = tf.python_io.TFRecordWriter(filename)
        duration = self.movie.duration # in seconds
        fps = self.movie.fps #
        frames = int(duration*fps)
        if cut is None:
            cut = range(frames)
        ## Collect all trajectory data:
        center,others = sorter(mouse)
        cent_traj = self.render_trajectory_full(center)
        parts_traj = self.render_trajectories(others)
        angles = -270-self.body_angle(mouse)*180./np.pi

        rel_traj = np.concatenate([part_traj-cent_traj for part_traj in parts_traj],axis = 1)

        for frameind in cut:
            imagedata = self.return_cropped_view_rot([mouse],frameind,angles[frameind])[0]
            image = imagedata
            # image = Image.fromarray(imagedata.astype(np.uint8))
            print(image.shape)
            pos = rel_traj[frameind,:]
            example = _write_ex_animal_focused(self.dataset_name,frameind,image,mouse,pos)
            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()

    def write_cropped_tfrecords(self,mouse,filename,cut= None):
        writer = tf.python_io.TFRecordWriter(filename)
        duration = self.movie.duration # in seconds
        fps = self.movie.fps #
        frames = int(duration*fps)
        if cut is None:
            cut = range(frames)
        ## Collect all trajectory data:
        center,others = sorter(mouse)
        cent_traj = self.render_trajectory_full(center)
        parts_traj = self.render_trajectories(others)


        rel_traj = np.concatenate([part_traj-cent_traj for part_traj in parts_traj],axis = 1)

        for frameind in cut:
            imagedata = self.return_cropped_view([mouse],frameind)[0]
            image = imagedata
            # image = Image.fromarray(imagedata.astype(np.uint8))
            print(image.shape)
            pos = rel_traj[frameind,:]
            example = _write_ex_animal_focused(self.dataset_name,frameind,image,mouse,pos)
            writer.write(example.SerializeToString())
        writer.close()
        sys.stdout.flush()

    # Return video of trajectories in polar coordinates relative to center:
    def render_trajectories_polar(self,start=0,stop=-1,to_render= None,save = False):
        ## Gaze angles:
        angles0 = self.gaze_relative(0)[1305:2820]
        angles1 = self.gaze_relative(1)[1305:2820]
        angles_together = [angles0,angles1]
        gaze0 = np.where(angles0< 0.1)[0]
        gaze1 = np.where(angles1< 0.1)[0]
        gaze0_ints = [[gaze0_val-1,gaze0_val+1] for gaze0_val in gaze0]
        gaze1_ints = [[gaze1_val-1,gaze1_val+1] for gaze1_val in gaze1]
        gaze_ints = [gaze0_ints,gaze1_ints]
        pindices = [3,8]
        rmax = 70
        all_thetas = []
        all_rs = []
        all_cents = []
        for mouse_number,pindex in enumerate(pindices):
            mouse_thetas = []
            mouse_rs = []
            if to_render == None:
                to_render = [index for index in self.part_index if index != pindex]

            reference_traj = self.render_trajectory_full(pindex)

            for other in to_render:
                part_traj = self.render_trajectory_full(other)
                relative = part_traj-reference_traj
                # Calculate r:
                rs = np.linalg.norm(relative,axis = 1)
                rs[rs>rmax] = rmax
                north = np.concatenate((np.ones((len(self.time_index),1)),np.zeros((len(self.time_index),1))),axis = 1)
                angles = angle_between_vec(north,relative)
                sign = np.sign(north[:,1]-relative[:,1])
                mouse_thetas.append(angles*sign)
                mouse_rs.append(rs)

            all_thetas.append(mouse_thetas)
            all_rs.append(mouse_rs)
        names = ['Virgin','Mother']
        colors = ['red','blue']
        for frame in range(1500):
            f = plt.figure(figsize = (20,20))
            ax0 = f.add_axes([0.35, 0.65, 0.25, 0.25], polar=True)
            ax1 = f.add_axes([0.7, 0.65, 0.25, 0.25], polar=True)
            ax = [ax0,ax1]
            ax_im0 = f.add_axes([0.35,0.35,0.25,0.25])
            ax_im1 = f.add_axes([0.7,0.35,0.25,0.25])
            ax_im = [ax_im0,ax_im1]
            ax_gaze = f.add_axes([0.35,0.1,0.6,0.20])
#             ax_full0 = f.add_axes([0.15,0.2,0.30,0.30])
            ax_full = f.add_axes([0.05,0.6,0.20,0.25])
            ax_enclosure = f.add_axes([0.05,0.2,0.20,0.30])
#             ax_full.axis('off')
#             ax_full = [ax_full0,ax_full1]
#             fig,ax = plt.subplots(2,2,subplot_kw=dict(projection='polar'))
#             slices = [np.range(9)[0:4],np.range(9)[5:9]]
            for mouse in range(2):
            ## Polar Plots:
                eff_mouse = [0]*(5-mouse)+[1]*(5-(1-mouse))
                for part in range(9):

                    ax[mouse].plot(all_thetas[mouse][part][start+frame:stop+frame],all_rs[mouse][part][start+frame:stop+frame],color = colors[eff_mouse[part]],linewidth = 2.,marker = 'o',markersize = 0.1)
                    ax[mouse].plot(all_thetas[mouse][part][stop+frame],all_rs[mouse][part][stop+frame],'o',color = colors[eff_mouse[part]])
                face_indices = mouse*4+np.array([0,1,2,0])

                ax[mouse].plot([all_thetas[mouse][face][stop+frame] for face in face_indices],[all_rs[mouse][face][stop+frame] for face in face_indices],color = colors[mouse],linewidth = 3.)
                body_index = mouse*4

                ax[mouse].plot([all_thetas[mouse][body_index][stop+frame],0],[all_rs[mouse][body_index][stop+frame],0],color = colors[mouse],linewidth = 3.)
                tail_index = mouse*5+3
                ax[mouse].plot([all_thetas[mouse][tail_index][stop+frame],0],[all_rs[mouse][tail_index][stop+frame],0],color = colors[mouse],linewidth = 3.)
                ax[mouse].set_rmax(rmax)
                ax[mouse].set_yticklabels([])
                ax[mouse].set_title(names[mouse]+' Centered Behavior',fontsize = 18)

                ## Image Plots:
                all_cents = self.render_trajectory_full(mouse*5+3)
                xcent,ycent = all_cents[stop+frame,0],all_cents[stop+frame,1]
                print(all_cents[mouse].shape)
        #         if (xcent < 550) and (ycent<400):
                print(xcent,ycent)
                radius = 50
                xsize,ysize = self.movie.size
                xmin,xmax,ymin,ymax = int(xcent-radius),int(xcent+radius),int(ycent-radius),int(ycent+radius)
                ## do edge detection:
                pads  = np.array([[ymin - 0,ysize - ymax],[xmin - 0,xsize - xmax],[0,0]])
                print(xmin,xmax,ymin,ymax)
                image = img_as_ubyte(self.movie.get_frame((stop+frame)/self.movie.fps))
                clip = image[ymin:ymax,xmin:xmax]
                if np.any(pads < 0):
            #         print('ehre')
                    topad = pads<0
                    padding = -1*pads*topad
                    clip = np.pad(clip,padding,'edge')

                ax_im[mouse].imshow(clip)
                ax_im[mouse].axis('off')
                ax_im[mouse].set_title(names[mouse]+' Zoomed View',fontsize = 18)
                ## Now do gaze detection:
                ax_gaze.plot(angles_together[mouse],color = colors[mouse],label = names[mouse])
#                 ax_gaze.plot(angles1,color = colors[mouse])
                ax_gaze.set_title('Relative Angle of Posture',fontsize = 18)
                ax_gaze.set_ylabel('Angle from Mouse Centroid (Absolute Radians)')
                ax_gaze.set_xlabel('Frame Number')
                ax_gaze.axvline(x = frame,color = 'black')
                ax_gaze.legend()
                for interval in gaze_ints[mouse]:
                    ax_gaze.axvspan(interval[0],interval[1],color =colors[mouse],alpha = 0.2)

                ax_full.plot(xcent,ycent,'o',color = colors[mouse])
                ax_full.set_yticklabels([])
                ax_full.set_xticklabels([])
                ax_full.plot(all_cents[start+frame-100:stop+frame,0],-all_cents[start+frame-100:stop+frame,1],color = colors[mouse],linewidth = 2.)
            ax_full.set_xlim(300,600)
            ax_full.set_ylim(-470,-70)
            ax_full.set_title('Physical Position',fontsize = 18)
            ax_enclosure.imshow(image[70:470,300:600])
            ax_enclosure.axis('off')
            ax_enclosure.set_title('Raw Video',fontsize = 18)
            plt.tight_layout()
            if save != False:
                plt.savefig('testframe%s.png'%frame)
                plt.close()
            else:
                plt.show()
                plt.close()
        if save != False:
            subprocess.call(['ffmpeg', '-framerate', str(10), '-i', 'testframe%s.png', '-r', '30','Angle_simmotion_'+'.mp4'])
#             for filename in glob.glob('testframe*.png'):
#                 os.remove(filename)
