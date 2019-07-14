import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from heapq import heappush,heappop
from timeit import default_timer as timer 
## A python module that holds the refactored code for behavioral trace analysis. This new code is based on the fundamental unit of a segment object, that is organized and held in a configuration object. This module should be independent of all the modules that have been written previously, but is testable using the SwitchCircle object in Social_Dataset_BB_testdata. 

## The fundamental unit of our new analysis. Contains segments of behavioral traces, chunked in time such that handling by higher order functions is simple and efficient. Carries a base, working, and query configuration to handle flexible working with multiple hypothetical  

invalid_int = -999999
## Helper function
## Handles indexing into numpy arrays using configuration arrays. 
def c2i(data,config):
    out = data[config]
    out[np.where(config == -1)] = invalid_int
    return out
## The same as c2i, but instead of converting to an index, converts to a float with nans as the default. 
def c2f(data,config):
    out = data[config]
    out[np.where(config == -1)] = np.nan 
    return out

    
## Setup functions: 
## Functions to get us from raw data to segments. 

## Function to define rules that get us from data to segments. In the future, we will want to integrate the process of generating these segmentations to work within an optimization framework as well. 
def make_blueprint():
    ## This can wait.  
    pass

## Function that takes trajectory, as well as time intervals and a mask. Returns a list of segment objects. 
def data_framer(trajectory,time,mask):
    ## Convert the mask to a configuration representation. 
    conf = mask.astype(int)*np.array([[1,2]])
    conf = conf-1
    segs = []
    for t,interval in enumerate(time):
        segind = t
        timeind = interval.astype(int)
        trace = trajectory[slice(*timeind),:]
        config = conf[t,:] 
        seg = Segment(trace,timeind,segind,config)
        segs.append(seg)
    return segs

class Segment(object):
    '''
    Object class to handle the low level nitty-gritty of working with behavioral trace data. Carries behavioral trace data, time and segment indices, and a local configuration set (base,working,query) to allow us to work with data flexibly.  
    
    Parameters: 
    trajectories: a numpy array of size (time,2*number of animals*number of parts). In reasonable cases for the forseeable future, probably will be (time,4). 
    timeindex: a tuple containing the start and end times occupied by the segment in the dataset of origin. 
    segindex: an integer, giving the ordinal index of the segment in the dataset of origin. 
    init_config: a two element list containing 1,0, or -1. These are codes for how to move around the trajectories to our desired initial configuration. 

    '''
    def __init__(self,trajectories,timeindex,segindex,init_config): 
        ## We have trajectory a and b: this is the way the trajectories are initialized. These attributes should be 100% immutable. 
        self.traja = trajectories[:,0:2]
        self.trajb = trajectories[:,2:4]
        self.trajs = np.stack((self.traja,self.trajb),axis=0)
        ## For indexing purposes: 
        self.timeindex = timeindex
        self.length = self.timeindex[-1]-self.timeindex[0]
        self.segindex = segindex
        ## Check for consistency between the attributes we have declared thus far.  
        assert len(self.trajs[0,:,:]) == self.timeindex[1]-self.timeindex[0],'timeindex and trajectories must match '
        ## Finally, initialize the configuration of the two trajectories: 
        self.base_config = init_config
        self.barred = np.where(self.base_config==-1)[0]
        self.allowed = np.where(self.base_config!=-1)[0]
        self.set_work_config(self.base_config)

        ## Initialize distance calculations, referenced later by other methods.  
        self.calculate_dist()
        self.calculate_tv()
        self.calculate_bounds()

    ## Configuration functions. 

    ## Checks if a given configuration is valid, given the base configuration that is known. If a configuration is -1 for a given segment index, another configuration cannot pull it, and it cannot be restored.   
    def check_config(self,config):
        disallowed = np.any([bi in config for bi in self.barred])
        return 1-disallowed

    ## Takes as input a numpy array of shape (1,number of individuals) 
    def set_work_config(self,config):
        cond = self.check_config(config)
        if cond == True:
            self.work_config = config
        else:
            print('Given config not compatible with segment '+str(self.segindex))

    ## Reset to the base config. 
    def reset_work_config(self):
        self.set_work_config(self.base_config)


    ## Measuring functions
    def calculate_dist(self): 
        '''
        Function to measure the distance from the point registered at the beginning and end of the trajectory. Registered to 'a' and 'b', will be handled later for specific configurations.  
        '''
       
        dista = np.linalg.norm(self.traja[-1,:]-self.traja[0,:])
        distb = np.linalg.norm(self.trajb[-1,:]-self.trajb[0,:])
        self.dists = np.array([dista,distb]) 
    
    def calculate_tv(self): 
        '''
        Function to measure the tv norm from the point registered at the beginning and end of the trajectory. Registered to 'a' and 'b', will be handled later for specific configurations.  
        '''
        tva = np.sum(np.linalg.norm(abs(np.diff(self.traja,axis=0)),axis = 1))
        tvb = np.sum(np.linalg.norm(abs(np.diff(self.trajb,axis=0)),axis = 1))
        self.tvs = np.array([tva,tvb])

    def calculate_bounds(self):
        '''
        Function to calculate start and end points (a,b, registered). 
        '''
        starts = np.array([self.traja[0,:],self.trajb[0,:]]) 
        ends = np.array([self.traja[-1,:],self.trajb[-1,:]]) 
        self.starts = starts
        self.ends = ends
        
    ## Functions that return things depending on the current configuration state. 
    def get_dist(self, query = None,mouse = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize distances according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                dists = c2f(self.dists,query) 
                
            else: 
                dists = np.array([np.nan,np.nan])
        else: 
            dists = c2f(self.dists,self.work_config) 
        ## Now index into the distances 
        if mouse is None: 
            dist= dists
        elif mouse in [0,1]:
            dist = dists[mouse]
        else:
            raise ValueError
        return dist
            
    def get_tv(self,query = None,mouse = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize tv according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                tvs = c2f(self.tvs,query) 
                
            else: 
                tvs = np.array([np.nan,np.nan])
        else: 
            tvs = c2f(self.tvs,self.work_config) 
        ## Now index into the tvances 
        if mouse is None: 
            tv= tvs
        elif mouse in [0,1]:
            tv = tvs[mouse]
        else:
            raise ValueError
        return tv
            
            
    def get_start(self, query = None,mouse = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize tv according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                starts = c2i(self.starts,query) 
            else: 
                starts = np.array([[invalid_int,invalid_int],[invalid_int,invalid_int]])
        else: 
            starts = c2i(self.starts,self.work_config) 
        ## Now index into the tvances 
        if mouse is None: 
            start = starts 
        elif mouse in [0,1]:
            start = starts[mouse]
        else:
            raise ValueError
        return start

    def get_end(self, query = None,mouse = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize tv according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                ends = c2i(self.ends,query) 
            else: 
                ends = np.array([[invalid_int,invalid_int],[invalid_int,invalid_int]])
        else: 
            ends = c2i(self.ends,self.work_config) 
        ## Now index into the tvances 
        if mouse is None: 
            end= ends
        elif mouse in [0,1]:
            end = ends[mouse]
        else:
            raise ValueError
        return end

    def get_traj(self, query = None,mouse = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, trajectory of both mice for working configuration will be returned.  
        '''
        ## Organize according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                trajs = c2f(self.trajs,query) 
            else: 
                trajs = np.nan*np.ones(np.shape(self.trajs))
        else: 
            trajs = c2f(self.trajs,self.work_config) 
        ## Now index into the correct trajectory: 
        if mouse is None: 
            traj= trajs
        elif mouse in [0,1]:
            traj = trajs[mouse]
        else:
            raise ValueError
        return traj
    
    def plot_trace():
        pass

## Trace traversal helper functions: 
## Find the first valid entry and return its index
def find_first(array):
    ## Add one to look for zeros: 
    plusarray = array+1
    relevant = next((i for i,x in enumerate(plusarray) if x),None)
    return relevant


## We define a configuration class that holds a list of segment objects, and can test various manipulations of them, and calculate the resulting costs. As with individual segments, the configuration object has different "tiers" of configuration, starting with the base configuration (no touch) to query configuration that are just passed to the data for the sake of testing out a particular configuration. The configuration code for Configurations has one more flag compared to the Segment: -2 says, we are agnostic, reference the current underlying trajectory. 
class Configuration(object):
    ## Lagrange multipliers are for the first and second penalty term when calculating the cost. 
    def __init__(self,trajectory,time,mask,lagrange1=None,lagrange2=None,mweights=None,sweights=None):
        self.segs = data_framer(trajectory,time,mask) 
        self.nb_segs = len(self.segs)
        self.length = np.sum([seg.length for seg in self.segs])
        self.lengths = np.array([self.segs[i].length for i in range(self.nb_segs)])
        self.base_config = np.stack([seg.base_config for seg in self.segs],axis = 0)
        self.set_work_config(self.base_config)
        self.calculate_timesegs()
        ## We need more: the starts and ends with respect to the base configuration. 
        self.all_starts = np.stack([self.segs[i].get_start() for i in range(self.nb_segs)],axis=1)
        self.all_ends = np.stack([self.segs[i].get_end() for i in range(self.nb_segs)],axis=1)
        self.all_tvs = np.stack([self.segs[i].get_tv() for i in range(self.nb_segs)],axis=0)

        ## If lagrange1 is not given, we can use the optimal with some default parameter: 
        if lagrange1 is None:
            lagrange1 = self.lagrange1_fromdata(0.05)
        if lagrange2 is None: 
            lagrange2 = 0.05
        self.weights = [lagrange1,lagrange2]
        if mweights is None:
            self.mweights,self.sweights = self.lagrange_moving(N = int(self.length//10))
        else:
            self.mweights = mweights
            self.sweights = sweights

    def check_config(self,config,start = None,end = None):
        assert len(config) == len(range(self.nb_segs)[start:end])
        ## Segmentwise check. First take the segment list, the relevant configs, and zip: 
        inputs = zip(self.segs[start:end],config)
        ## Now broadcast with map. 
        checks = list(map(lambda x: Segment.check_config(*x),inputs))
        ## The configuration is only valid if all the entires are individually valid. 
        check = np.all(checks)
        return check
    
    def return_checked(self,query,start = None,end = None):
        if query is None:
            config = self.work_config[start:end]
        else:
            assert self.check_config(query,start = start,end = end); 'Must be valid'
            config = np.copy(query)
            if -2 in config:
                config[np.where(config ==-2)] = self.work_config[start:end][np.where(config==-2)]
        return config

    def set_work_config(self,config):
        cond = self.check_config(config)
        if cond == True:
            ## First take the agnostic parts, and replace with working config: 
            if -2 in config:
                config[np.where(config ==-2)] = self.work_config[np.where(config==-2)]
            self.work_config = config
            ## We also have to change it in the underlying segments! 
            inputs = zip(self.segs,config)
            list(map(lambda x: Segment.set_work_config(*x),inputs))
        else:
            print('Given config not compatible with configuration')

    def reset_work_config(self):
        self.set_work_config(self.base_config)

    ## Now, some functions to allow for easy indexing. Let's see if we can pull out a trajectory segment by timepoints. 
    ## First, a helper function to give the segment index of any single time point. 
    def calculate_timesegs(self):
        indparts = []
        for seg in self.segs: 
            indpart = np.repeat(seg.segindex,seg.length)
            
            indparts.append(indpart)
        timesegs = np.concatenate(indparts)
        self.timesegs = timesegs

    ## Pull out the positions of the two mice at any given query point. Query in this case should just be a tuple. Useful for estimating costs.   
    def render_point_time(self,time,query = None):
        ## First get out the starting and ending segments. 
        segind = self.timesegs[time]
        seg = self.segs[segind]
        ## Initialize the trajectory to be rendered as an array of nans: 
        ## Specify the configuration we will use to generate this trajectory
        if query is None:
            config = self.work_config
        else:
            assert self.check_config(query); 'Must be valid'
            config = query
        ## Now get the trajectory: 
        trajectory = seg.get_traj(query = config[segind])
        ## Shift the given time index to match the trajectory: 
        index_inseg = time-seg.timeindex[0]
        point = trajectory[:,index_inseg,:]
        return point 

    ## Renders the trajectory of both animals between the given time indices. Time indices are interpreted slice-style, with the last one non-inclusive. 
    def render_trajectory_time(self,timeindex = None,query = None):
        ## Default timeindex is the whole trajectory
        if timeindex is None: 
            timeindex = np.array([0,self.length])
        s,e = timeindex[0],timeindex[-1]-1
        ## First get out the starting and ending segments. 
        start,end = self.timesegs[np.array([s,e])]
        ## Initialize the trajectory to be rendered as an array of nans: 
        trajectory = np.ones((2,timeindex[-1]-timeindex[0],2))*np.nan
        ## Specify the configuration we will use to generate this trajectory
        if query is None:
            config = self.work_config[slice(start,end+1),:]
        else:
            config = self.return_checked(query,start,end+1) 
        for sord,segind in enumerate(np.arange(start,end+1)):
            ## Get the trajectory as appropriate: 
            seg = self.segs[segind]
            segtrajectory = seg.get_traj(query = config[sord])
            if segind == start: 
                segstart = timeindex[0]-seg.timeindex[0]
            else: 
                segstart = 0
            if segind == end: 
                segend = timeindex[-1]-seg.timeindex[0]
            else:
                segend = seg.length
            ## Grab the time indices
            segtimes = seg.timeindex 
            ## Shift the time indices to respect the start of the trajectory and turn them into a slice. Also handle cases where the indices overlap. 
            shiftindex = segtimes-s
            shiftindex[np.where(shiftindex<0)] = 0
            segtimes_slice = slice(*shiftindex)
            ## Now assign. 
            trajectory[:,segtimes_slice,:] = segtrajectory[:,segstart:segend,:]
        return trajectory 

    def render_trajectory_seg(self,segindex = None,query = None):
        ## Default timeindex is the whole trajectory
        if segindex is None: 
            tstart,tend = 0,self.length
        else:
            tstart,tend = self.segs[segindex[0]].timeindex[0],self.segs[segindex[-1]].timeindex[-1]
        trajectory = self.render_trajectory_time(np.array([tstart,tend]),query)
        return trajectory 

    ## TODO: Plotting functions: compare to check_plot

    ## Cost functions: 
    ## In order to define cost functions, we have to have a notion of valid segments. Have a function that will return two lists of the valid segment numbers under the current configuration. 
    ## TODO: vaildate get_valid0 against the whole trajectory. 
    def get_valid(self,query = None,start = None,end = None,check= True): 
        if check == True:
            config = self.return_checked(query,start=start,end =end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        ## Start with the working configuration: 
        lengtharray = np.arange(self.length)[start:end]
        length= len(lengtharray)# booo
        indexarray = lengtharray.reshape(length,1).repeat(2,axis = 1)
        valindices = []
        if start is None:
            startind = 0
        else:
            startind = start
        for i in range(2):
            valid = indexarray[:,i][np.where(config[:,i]!=-1)]-startind
            valindices.append(valid)
        return valindices

    ## Get the valid segment indices that flank this point. 
    ## We have to do some low level stuff here: 
    def get_valid_surr(self,segindex,mouse,query = None,check = True):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        prepoints = []
        postpoints = []
        mouseconfig = config[:,mouse]
        mousepre,mousepost = np.flip(mouseconfig[:segindex]),mouseconfig[segindex:]
        preind,postind = [find_first(m) for m in [mousepre,mousepost]]
        if preind is None:
            backind = None 
        else:
            backind = segindex-(preind+1)
        if postind is None:
            forind = None 
        else:
            forind = segindex+postind
        return backind,forind
            
    ## The easier part is getting the inter segment intervals for all valid intervals.
    ## This function returns the array of all valid intra segment costs (excluding nans) under the given (or working) configuration. 
    def get_intra(self,query = None,check = True):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        valids = self.get_valid(config,check = False)
        mice_ids = [config[valids[i],i] for i in range(2)]
        tvs = np.ones(self.all_tvs.shape)*np.nan
        for i in range(2):
            tvs[valids[i],i] = self.all_tvs[valids[i],mice_ids[i]]
        return tvs

    ## Deprecated. here for reference in case of fires
    #def get_intra(self,query = None,check = True):
    #    if check == True:
    #        config = self.return_checked(query)
    #    else:
    #        if query is None:
    #            config = self.work_config
    #        else:
    #            config = query
    #    inputs = zip(self.segs,config)
    #    tv = np.stack([self.segs[i].get_tv(config[i]) for i in range(self.nb_segs)])
    #    return tv
    
    def get_intra_cut(self,query = None,start = None,end = None,check = True):
        if check == True:
            config = self.return_checked(query,start = start,end = end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        ## Returns aligned to the start provided for indexing into the config provided. 
        valids = self.get_valid(config,start = start,end = end,check = False)
        tvs = np.ones(self.all_tvs[start:end].shape)*np.nan
        for i in range(2):
            if len(valids[i])>0:
                mice_ids = config[valids[i],i]
                tvs[valids[i],i] = self.all_tvs[valids[i],mice_ids]
        return tvs
    
    #def get_intra_cut(self,query = None,start = None,end = None,check = True):
    #    if check == True:
    #        config = self.return_checked(query,start = start,end = end)
    #    else:
    #        if query is None:
    #            config = self.work_config[start:end]
    #        else:
    #            config = query
    #    inputs = zip(self.segs[start:end],config)
    #    tv = np.stack(list(map(lambda x: Segment.get_tv(*x),inputs)))
    #    return tv
    
    def get_intra_dist(self,query = None):
        config = self.return_checked(query)
        inputs = zip(self.segs,config)
        tv = np.stack(list(map(lambda x: Segment.get_dist(*x),inputs)))
        return tv

    def get_intra_dist_cut(self,query = None,start = None,end = None,check = True):
        if check == True:
            config = self.return_checked(query,start = start,end = end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        inputs = zip(self.segs[start:end],config)
        dist = np.stack(list(map(lambda x: Segment.get_dist(*x),inputs)))
        return dist 

    ## The slightly harder part: this function returns two lists of all valid inter segment costs under the given (default working) configuration. Returns an array of shape (number_intervals-1,2). If the gap spans multiple invalid trajectories, put the actual cost in the first interval for which the whole cost applies.  
    ## A version that does not have handling of these different cases: 
    def get_inter(self,query = None,check = True):    
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        valid_inds = self.get_valid(config,check = False)
        # Now iterate over each mouse: 
        dist_length = np.max([len(config)-1,1])
        dists  = np.zeros((dist_length,2)) 
        ## Get the mouse indices corresponding to the time indices: 
        mice_inds = [config[v,vi] for vi,v in enumerate(valid_inds)]
        for m in range(2):
            starts = self.all_starts[mice_inds[m][1:],valid_inds[m][1:]]
            ends = self.all_ends[mice_inds[m][:-1],valid_inds[m][:-1]]
            dist_all = np.linalg.norm(ends-starts,axis = 1)
            dists[valid_inds[m][:-1],m] = dist_all
            if len(valid_inds[m]) == 0:
                endtime = 0
                starttime = self.length
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                dists[0,m] = tv_opt 
            elif valid_inds[m][0] != 0:
                endtime = 0
                starttime = self.segs[valid_inds[m][0]].timeindex[0]
                ## Query the moving average weights we collected, and be optimistic:  
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                dists[0,m] = tv_opt 
            elif valid_inds[m][-1] != self.nb_segs-1:
                endtime = self.segs[valid_inds[m][-1]].timeindex[-1]
                starttime = self.length
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                dists[-1,m] = tv_opt 
        
        return dists

    #def get_inter(self,query = None,check = True):    
    #    if check == True:
    #        config = self.return_checked(query)
    #    else:
    #        if query is None:
    #            config = self.work_config
    #        else:
    #            config = query
    #    valid_inds = self.get_valid(config,check = False)
    #    # Now iterate over each mouse: 
    #    dist_length = np.max([len(config)-1,1])
    #    dists  = np.zeros((dist_length,2)) 
    #    for m in range(2):
    #        mouse_valid = valid_inds[m]
    #        ## Now we would like to identify all pairs: 
    #        pairs = [mouse_valid[i:i+2] for i in range(len(mouse_valid)-1)]
    #        for pair in pairs: 
    #            end = self.segs[pair[0]].get_end(mouse = m,query = config[pair[0]])
    #            start = self.segs[pair[-1]].get_start(mouse = m,query = config[pair[-1]])
    #            dist = np.linalg.norm(end-start)
    #            dists[pair[0],m] = dist
    #        if len(mouse_valid) == 0:
    #            endtime = 0
    #            starttime = self.length
    #            tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
    #            tv_opt = np.sum(tvs)
    #            dists[0,m] = tv_opt 
    #        elif mouse_valid[0] != 0:
    #            endtime = 0
    #            starttime = self.segs[mouse_valid[0]].timeindex[0]
    #            ## Query the moving average weights we collected, and be optimistic:  
    #            tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
    #            tv_opt = np.sum(tvs)
    #            dists[0,m] = tv_opt 
    #        elif mouse_valid[-1] != self.nb_segs-1:
    #            endtime = self.segs[mouse_valid[-1]].timeindex[-1]
    #            starttime = self.length
    #            tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
    #            tv_opt = np.sum(tvs)
    #            dists[-1,m] = tv_opt 
    #    
    #    return dists

    def get_inter_cut(self,query = None,start = None,end = None,check = True):
        if check == True:
            config = self.return_checked(query,start,end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        if start is None:
            startp = 0
        else:
            startp = start
        ## Returns aligned to the start provided.
        valid_inds = self.get_valid(config,start,end,check = False)
        # Now iterate over each mouse: 
        dist_length = np.max([len(config)-1,1])
        dists  = np.zeros((dist_length,2)) 
        ## Get the mouse indices corresponding to the time indices: 
        for m in range(2):
            if len(valid_inds[m])>0:
                mice_inds = config[valid_inds[m],m]
                starts = self.all_starts[mice_inds[1:],valid_inds[m][1:]+startp]
                ends = self.all_ends[mice_inds[:-1],valid_inds[m][:-1]+startp]
                dist_all = np.linalg.norm(ends-starts,axis = 1)
                dists[valid_inds[m][:-1],m] = dist_all
        return dists

    #def get_inter_cut(self,query = None,start = None,end = None,check = True):
    #    if check == True:
    #        config = self.return_checked(query,start,end)
    #    else:
    #        if query is None:
    #            config = self.work_config[start:end]
    #        else:
    #            config = query
    #    if start is None:
    #        startp = 0
    #    else:
    #        startp = start
    #    valid_inds = self.get_valid(config,start,end,check = False)
    #    # Now iterate over each mouse: 
    #    dist_length = np.max([len(config)-1,1])
    #    dists  = np.zeros((dist_length,2)) 
    #    for m in range(2):
    #        mouse_valid = valid_inds[m]
    #        ## Now we would like to identify all pairs: 
    #        pairs = [mouse_valid[i:i+2] for i in range(len(mouse_valid)-1)]
    #        for pair in pairs: 
    #            endpos = self.segs[pair[0]].get_end(mouse = m,query = config[pair[0]-startp])
    #            startpos = self.segs[pair[-1]].get_start(mouse = m,query = config[pair[-1]-startp])
    #            dist = np.linalg.norm(endpos-startpos)
    #            dists[pair[0]-startp,m] = dist
    #    return dists

    def get_inter_optimal(self,query = None,check = True):    
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        valid_inds = self.get_valid(config,check = check)
        # Now iterate over each mouse: 
        length_tv = np.max([len(config)-1,1])
        tv_opts = np.zeros((length_tv,2)) 
        for m in range(2):
            mouse_valid = valid_inds[m]
            ## Now we would like to identify all pairs: 
            pairs = [mouse_valid[i:i+2] for i in range(len(mouse_valid)-1)]
            for pair in pairs: 
                endtime = self.segs[pair[0]].timeindex[-1]-1
                starttime = self.segs[pair[-1]].timeindex[0]  
                ## Query the moving average weights we collected, and be optimistic:  
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                tv_opts[pair[0],m] = tv_opt 
            if len(mouse_valid) == 0:
                endtime = 0
                starttime = self.length
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                dists[0,m] = tv_opt 
            elif mouse_valid[0] != 0:
                endtime = 0
                starttime = self.segs[mouse_valid[0]].timeindex[0]
                ## Query the moving average weights we collected, and be optimistic:  
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                dists[0,m] = tv_opt 
            elif mouse_valid[-1] != len(tv_opts):
                endtime = self.segs[mouse_valid[-1]].timeindex[-1]-1
                starttime = self.length 
                tvs = self.mweights[endtime:starttime,m]-self.sweights[endtime:starttime,m]
                tv_opt = np.sum(tvs)
                tv_opts[-1,m] = tv_opt 
        return tv_opts
    ## Get full cost. Takes a split argument that partitions the cost by the segment index. Splits up to the start of the indicated segment. 

    def full_tvcost(self,query = None,check = True):
        ## The intra segment cost: 
        tvarray = self.get_intra(query,check = check)
        distarray = self.get_inter(query,check = check)
        cost = np.nansum(tvarray)+np.sum(distarray)
        return cost 

    ## Test this on an extra long example
    ## Get the cost split at an indicated segment. 
    def split_tvcost(self,query = None,split=0,check = True):
        ## The intra segment cost: 
        tvarray = self.get_intra(query,check = check)
        distarray = self.get_inter(query,check = check)
        pretv,posttv = tvarray[:split,:],tvarray[split:,:]
        predist,postdist = distarray[:split,:],distarray[split:,:] 
        precost = np.nansum(pretv)+np.sum(predist)
        postcost = np.nansum(posttv)+np.sum(postdist)
        return precost,postcost 

    ## Culmination of efficient cost calculation functions. 
    def cut_tvcost(self,query = None,start = None,end = None,check = True):
        ## First calculate all intra costs: 
        ## Handle corner cases: 
        if start == self.nb_segs or end == 0:
            cost = 0
        else:
            intraarray = self.get_intra_cut(query,start,end,check)
            interarray = self.get_inter_cut(query,start,end,check)
            cost = np.nansum(intraarray)+np.nansum(interarray)
        return cost

    ## Write pre/post split functions that will actually save time by computing only partial costs 

    ## Get the penalty due to regularization terms: 
    ## The cost for removing datapoints:
    def full_nullcost(self,query=None,check = True):
        ## Find all segments that have negative ones
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config ==-1)] = 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths*counts*self.weights[0]
        return np.sum(costvec)
    
    def split_nullcost(self,query=None,split = 0,check = True):
        ## Find all segments that have negative ones
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config ==-1)] = 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths*counts*self.weights[0]
        return np.sum(costvec[:split]),np.sum(costvec[split:])
        
    def cut_nullcost(self,query=None,start=None,end=None,check = True):
        ## Find all segments that have negative ones
        if check == True:
            config = self.return_checked(query,start,end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config ==-1)] = 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths[start:end]*counts*self.weights[0]
        return np.sum(costvec)

    def full_switchcost(self,query=None,check = True):
        ## Find all segments that have flipped indices relative to the base configuration we started with. 
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        ## Compare against an entirely flipped array: 
        fliparray = np.zeros(np.shape(config))
        fliparray[:,0] += 1
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config == fliparray)] == 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths*counts*self.weights[1]
        return np.sum(costvec)

    def split_switchcost(self,query = None,split = 0,check = True):
        ## Find all segments that have flipped indices relative to the base configuration we started with. 
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        ## Compare against an entirely flipped array: 
        fliparray = np.zeros(np.shape(config))
        fliparray[:,0] += 1
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config == fliparray)] == 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths*counts*self.weights[1]
        return np.sum(costvec[:split]),np.sum(costvec[split:])
    
    def cut_switchcost(self,query = None,start = None,end = None,check = True):
        ## Find all segments that have flipped indices relative to the base configuration we started with. 
        if check == True:
            config = self.return_checked(query,start,end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        ## We would like an efficient way to count lengths: have an integer count of how many individuals designate a given part as invalid. 
        ## Compare against an entirely flipped array: 
        fliparray = np.zeros(np.shape(config))
        fliparray[:,0] += 1
        indicator = np.zeros(np.shape(config))
        indicator[np.where(config == fliparray)] = 1
        counts = np.sum(indicator,axis = 1)
        costvec = self.lengths[start:end]*counts*self.weights[1]
        return np.sum(costvec)

    def full_cost(self,query=None,check = True):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        tvcost = self.full_tvcost(query=config,check = False)
        p1cost = self.full_nullcost(query=config,check = False)
        p2cost = self.full_switchcost(query=config,check = False)
        return tvcost+p1cost+p2cost

    def split_cost(self,query=None,split = 0,check = True):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        tvcost0,tvcost1 = self.split_tvcost(query=config,split = split,check = False)
        p1cost0,p1cost1 = self.split_nullcost(query=config,split = split,check = False)
        p2cost0,p2cost1 = self.split_switchcost(query=config,split = split,check = False)
        return np.sum(np.stack((tvcost0,p1cost0,p2cost0)),axis = 0),np.sum(np.stack((tvcost1,p1cost1,p2cost1)),axis = 0)

    def cut_cost(self,query = None, start = None,end = None,check = True):
        if check == True:
            config = self.return_checked(query,start,end)
        else:
            if query is None:
                config = self.work_config[start:end]
            else:
                config = query
        cut_tv = self.cut_tvcost(query=config,start=start,end=end,check=False)
        cut_p1 = self.cut_nullcost(query=config,start=start,end=end,check=False)
        cut_p2 = self.cut_switchcost(query=config,start=start,end=end,check=False)
        return cut_tv+cut_p1+cut_p2

    def bridge_cost(self,query = None,split = 0,check = True,inds = [0,1]):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        additional_cost = 0
        for ind in inds:
            end,start = self.get_valid_surr(split,ind,config,False)
            if end is None:
                endpos = self.segs[0].get_start(mouse = ind,query = config[0])
            else:
                endpos = self.segs[end].get_end(mouse = ind,query = config[end])
            if start is None:
                startpos = self.segs[-1].get_end(mouse = ind,query = config[-1]) 
            else:
                startpos = self.segs[start].get_start(mouse = ind,query = config[start])
            #if start is None end is None: 
            #    pass
            #else:
            dist = np.linalg.norm(startpos-endpos)
            additional_cost+=dist
        return additional_cost

    def bridge_cost_opt(self,query = None,split = 0,check = True,inds = [0,1]):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        additional_cost = 0
        for ind in inds:
            end,start = self.get_valid_surr(split,ind,config,False)
            if end is None:
                endtime = self.segs[0].timeindex[0] 
            else:
                endtime= self.segs[end].timeindex[-1]
            if start is None:
                starttime = self.segs[-1].timeindex[-1]
            else:
                starttime = self.segs[start].timeindex[0]
            #if start is None end is None: 
            #    pass
            #else:
            tvs = self.mweights[endtime:starttime,ind]-self.sweights[endtime:starttime,ind]
            tvopt = np.nansum(tvs)
            additional_cost+=tvopt
        return additional_cost

    def bridge_adaptive(self,prolif_start,prolif_end,query = None, check = True):
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        ## Now, check if the segment is valid under this query:  
        valcheck = config[prolif_start:prolif_end]
        for i in range(2): 
            vals = valcheck[:,i]
            if np.all(vals == -1):
                bridge = self.bridge_cost_opt(query = config, split = prolif_end,inds = [i],check = check) 
            else:
                bridgepre = self.bridge_cost(query = config, split = prolif_start,inds = [i],check = check) 
                ## Should be beginning of estimation: use the optimal.
                bridgepost = self.bridge_cost_opt(query = config, split = prolif_end,inds = [i],check = check) 
                bridge = bridgepre+bridgepost
        return bridge

        

    ## Basically a test function to check the validity of what we're doing. 
    def split_efficient(self,query = None,split = 0,check = True):
        ## We have to specifically handle the case where the split happens across an invalid segment.  
        if check == True:
            config = self.return_checked(query)
        else:
            if query is None:
                config = self.work_config
            else:
                config = query
        additional_cost = self.bridge_cost(config,split = split,check = False)
        precost = self.cut_cost(config[:split],end = split,check = False)
        postcost = self.cut_cost(config[split:],start = split,check = False)

        return precost+additional_cost,postcost 

    ## Purely for the convenience of not having to index:
    def pre_cost(self,query=None,split = 0,check = True):
        tvcost0,tvcost1 = self.split_tvcost(query,split = split,check = check)
        p1cost0,p1cost1 = self.split_nullcost(query,split = split,check = False)
        p2cost0,p2cost1 = self.split_switchcost(query,split = split,check = False)
        return np.sum(np.stack((tvcost0,p1cost0,p2cost0)),axis = 0)

    ## Mini function to estimate distances: 

    def estimate_distance(self,nbp,config,split):
        timepoints = np.sort(np.random.choice(np.arange(split,self.length),nbp,replace = False))
        full_timepoints = np.concatenate([[split],timepoints])
        pointgetter = lambda t: self.render_point_time(t,query = config)
        points = np.stack(list(map(pointgetter,full_timepoints)))
        ## Add on the point given by the split position: 
        
        ## Get two arrays, one for each mouse, without nans: 
        m1p,m2p = points[:,0,:],points[:,1,:]
        m1clean,m2clean = m1p[~np.isnan(m1p)].reshape(-1,2),m2p[~np.isnan(m2p)].reshape(-1,2)
        m1diff,m2diff = np.diff(m1clean,axis = 0),np.diff(m2clean,axis = 0)
        m1norm,m2norm = np.linalg.norm(m1diff,axis=1),np.linalg.norm(m2diff,axis=1) 
        m1dist,m2dist = sum(m1norm),sum(m2norm)
        return m1dist+m2dist

    ## Estimate the cost under the given (or current) configuration, using an estimate of the tv per time rate. 
    def estimate_postcost(self,nbp,query=None,split = 0,check = True): 
        if check == True:
            config = self.return_checked(query)
        else:
            config = query
        ## Get out the tv lengths of the segments: 
        tvs = self.get_intra(config)
        tv = self.get_intra()
        ## Get out the corresponding distances: 
        dist = self.get_intra_dist()
        ## Ratio:
        ratio = tv/dist
        meanrate = np.nanmean(ratio)
        ## We will use this mean rate to calculate an estimate of the tv distance traveled. 
        estdist = self.estimate_distance(nbp,query,split)
        ## Calculate the distance*meanrate calculated above. 
        est_tvdist = meanrate*(estdist)
        return est_tvdist 
    
    def estimate_cost(self,nbp,copies = 1,query = None,split = 0,check = True):
        pre = self.pre_cost(query = query,split = split,check = check)
        post = np.array([self.estimate_postcost(nbp,query = query,split = split,check = False) for i in range(copies)])
        p1cost = self.full_nullcost(query)
        p2cost = self.full_switchcost(query)
        return pre+post+p1cost+p2cost

    ## Estimates the optimal cost past a certain point. Is NOT query dependent. 
    def estimate_optimal_post(self,start = 0):
        ## First, get all of the intra costs:
        tvs = self.get_intra()
        ## In addition, let's get the optimal inter segment distances, as projected using the tv norm: 
        inter_opt = self.get_inter_optimal()
        inter_opt_f = np.concatenate((inter_opt,np.array([[0,0]])),axis = 0)
        return tvs[start:]+inter_opt_f[start:]

    ## Rough tests done on the above. Merits more careful testing. 
    ## Calculate the optimal regularization terms (self.weights,this is given in your notebook) 
    ## We can fit the first lagrange parameter by looking at characteristics of the trajectory we know to be representative. As of right now, this works only drawing upon the working configuration. 
    def lagrange1_fromdata(self,p,plot = False):
        ## We want to calculate the normalized contour integral, and the normalized distances across all valid segments. 
        ## First get all lengths: 
        lengths = np.array([self.segs[i].length for i in range(len(self.segs))])[:,None]
        ## Now get all contour lengths, i.e. tv norms. 
        tvs = np.stack([self.segs[i].get_tv() for i in range(len(self.segs))])
        dists = np.stack([self.segs[i].get_dist() for i in range(len(self.segs))])
        ntvs = tvs/lengths
        ndists = dists/lengths
        ## Now we have an empirical distribution over realistic normed distances: 
        distrib = ntvs-ndists
        vals = distrib[~np.isnan(distrib)]
        ## Plot the valid values: 

        ## Our expectation is that the TV distance will look less reasonable compared to the distance travelled when a jump occurs. This leads to a distribution on R+ of trajectories we expect from real trajectories, encouraging us to reweight certain aspects of the cost.  
        ## Calculate MLE of an exponential from this distribution: 
        lambda_mle = 1./np.nanmean(distrib)
        if plot == True: 
            plt.hist(vals,bins=100,density = True)
            count = np.linspace(0,0.02,100)
            plt.plot(count,lambda_mle*np.exp(-lambda_mle*count))
            plt.show()
        ## We now take our cutoff p and generate the cutoff lagrange parameter we should take in order to satisfy this cutoff 1 using the expression for the cdf of an exponential.  
        xcutoff = -np.log(p)/lambda_mle
        return xcutoff

    ## we don't really know how to learn this from data right now... 
    def lagrange2_fromdata(self,p):
        pass

    ## This is a moving average lagrange function. 
    def lagrange_moving(self,N):
        ## First, we will fill in an array of tv norms for the trajectory, one segment at a time. 
        tvarray = np.ones((self.length,2))*np.nan
        for m in range(2):
            for valid in self.get_valid()[m]:
                seg = self.segs[valid]
                ## get endpoints: 
                start,end = seg.timeindex
                tv = np.linalg.norm(abs(np.diff(seg.trajs,axis = 1)),axis = 2)[m,:]
                ## Fill in our array (will be one shorter)
                tvarray[start:end-1,m] = tv 
        ## Now, we will calculate a moving average and moving standard deviation over this using a deque. 
        temp0 = deque([],maxlen = N)
        temp1 = deque([],maxlen = N)
        temp = [temp0,temp1]
        means = [[],[]]
        stds = [[],[]]
        for i in tqdm(range(len(tvarray)-N)):
            for j in range(2):
                ent = tvarray[i,j]
                temp[j].append(ent)
                mean = np.nanmean(temp[j])
                std = np.nanstd(temp[j])
                means[j].append(mean)
                stds[j].append(std)
        meanarray = np.array(means).T
        stdarray = np.array(stds).T
        ## We are now going to do linear interpolation where we have only nans: 
        meanis = []
        stdis = []
        for m in range(2):
            means,stds = meanarray[:,m],stdarray[:,m]
            valid = np.where(~np.isnan(means))
            if len(valid[0])>0:
                ## Interpolation indices:
                interidx = valid[0]+N//2
                ## interpolation points: 
                meanv,stdv = means[valid],stds[valid]
                evalrange = np.arange(self.length)
                
                meani,stdi = np.interp(evalrange,interidx,meanv),np.interp(evalrange,interidx,stdv)
            else:
                meani,stdi = meanv,stdv
            meanis.append(meani)
            stdis.append(stdi)
        meanfinal = np.stack(meanis).T
        stdfinal = np.stack(stdis).T
        return meanfinal,stdfinal

    ## Plot the cost per segment. 
    def plot_intra_cost(self,query = None,highlight = [None]):
        ## We want to plot each segment by its length. 
        config = self.return_checked(query)
        fig,ax = plt.subplots(2,1)
        for s,seg in enumerate(self.segs):
            start,end = seg.timeindex
            ## Now plot for each mouse:
            for i in range(2):
                ## Check if the segment is valid:   
                linestyle = ':'
                marker = '.'
                ax[i].plot(np.linspace(start,end,seg.length),np.ones(seg.length)*seg.get_tv(query = config[s],mouse = i),linestyle = linestyle,marker = marker,markersize = 1) 
                ## Highlight certain segments if we wish to: 
                if s in highlight: 
                    [ax[i].axvline(x = p,color = 'red',linestyle = '--') for p in seg.timeindex]
        ## Do labels and titles: 
        ax[1].set_xlabel('time (frames)')
        ax[0].set_ylabel('cost (tv norm)')
        ax[0].set_title('Object 1 Costs')
        ax[1].set_title('Object 2 Costs')
        plt.tight_layout()
        plt.show()

    ## Plot the full cost in terms of segment by segment contributions. 
    def plot_full_cost(self,query = None,segindex = None,highlight = [None]):
        config = self.return_checked(query)
        fig,ax = plt.subplots(2,1,sharex = True)
        ## Determine the range you want to plot across: 
        if segindex is None: 
            segindex = [0,len(self.segs)]
        for mouse in range(2):
            ## Just plot the trajectories of the valid segments. 
            prevalid = self.get_valid(query=config)[mouse]
            valid = prevalid[np.where(np.logical_and(prevalid<segindex[-1],prevalid>=segindex[0]))]
            ## Now plot each trajectory:
            for v,valind in enumerate(valid): 
                ax[mouse].plot(np.arange(*self.segs[valind].timeindex),self.render_trajectory_seg(segindex = [valind,valind],query = config[valind:valind+1])[mouse,:,0],'b')
                ax[mouse].plot(np.arange(*self.segs[valind].timeindex),self.render_trajectory_seg(segindex = [valind,valind],query = config[valind:valind+1])[mouse,:,1],'r')
                if v in highlight: 
                    [ax[mouse].axvline(x = p,color = 'red',linestyle = '--') for p in self.segs[valind].timeindex]
            ## Now plot the gaps between trajectories. 
            valpairs = [valid[i:i+2] for i in range(len(valid)-1)] 
            for valpair in valpairs:
                end = self.segs[valpair[0]].get_end(mouse = mouse,query = config[valpair[0]])
                start = self.segs[valpair[-1]].get_start(mouse = mouse,query = config[valpair[-1]])
                endtime = self.segs[valpair[0]].timeindex[-1]
                starttime = self.segs[valpair[-1]].timeindex[0]
                ax[mouse].plot([endtime,starttime],[end,start],'h:',color = 'black',markersize = 3)
        ax[0].set_ylabel('trajectory')
        ax[1].set_xlabel('time (frames)')
        ax[0].set_title('Object 1 trajectory')
        ax[1].set_title('Object 2 trajectory')
        plt.tight_layout()
        plt.show()

## Configuration object helper functions:
## Helper functions for helper functions: 
def convert_single(integer):
    assert integer < 3 and integer > -1
    #if integer == 0:
    #    integer -=1
    return integer-1

def convert_double(integer):
    assert integer < 9 and integer > -1
    tuple_raw = np.array([integer%3-1,integer//3-1])
    #tuple_raw[np.where(tuple_raw == -1)] =-2
    return tuple_raw[0],tuple_raw[1]


## Convert from signature to query configuration, with left entry first. 

def s2q_left(signature,length):
    ## Assume we take in a list. 
    converted = list(map(convert_single,signature))
    if len(converted)%2 == 1: 
        converted.append(-2)
    array = np.array(converted).reshape(-1,2)
    filler = np.ones((length-len(array),2))*-2
    fullarray = np.concatenate((array,filler),axis = 0)
    return fullarray.astype(int) 

def s2q_left_default(signature,length,default):
    ## Assume we take in a list. 
    converted = list(map(convert_single,signature))
    sig_length = np.ceil(len(converted)/2.).astype(int)
    if len(converted)%2 == 1: 
        converted.append(default[sig_length,-1])
    array = np.array(converted).reshape(-1,2)
    filler = default[sig_length:] 
    fullarray = np.concatenate((array,filler),axis = 0)
    assert length == len(fullarray)
    return fullarray.astype(int) 
    
def s2q_right(signature,length):
    ## Assume we take in a list. 
    converted = list(map(convert_single,signature))
    if len(converted)%2 == 1: 
        converted = converted[:-1]+[-2,converted[-1]]
    array = np.array(converted).reshape(-1,2)
    filler = np.ones((length-len(array),2))*-2
    fullarray = np.concatenate((array,filler),axis = 0)
    return fullarray.astype(int) 

def s2q_right_default(signature,length,default):
    ## Assume we take in a list. 
    converted = list(map(convert_single,signature))
    sig_length = np.ceil(len(converted)/2.).astype(int)
    if len(converted)%2 == 1: 
        converted = converted[:-1]+[default[sig_length,-1],converted[-1]]
    array = np.array(converted).reshape(-1,2)
    filler = default[sig_length:]
    fullarray = np.concatenate((array,filler),axis = 0)
    assert length == len(fullarray)
    return fullarray.astype(int) 

def s2q_both(signature,length):
    ## Assume we take in a list. 
    converted = list(map(convert_double,signature))
    array = np.array(converted).reshape(-1,2)
    filler = np.ones((length-len(array),2))*-2
    fullarray = np.concatenate((array,filler),axis = 0)
    return fullarray.astype(int) 

def s2q_both_default(signature,length,default):
    ## Assume we take in a list. 
    converted = list(map(convert_double,signature))
    sig_length = len(converted)
    array = np.array(converted).reshape(-1,2)
    filler = default[sig_length:]
    fullarray = np.concatenate((array,filler),axis = 0)
    assert length == len(fullarray)
    return fullarray.astype(int) 

def allowed2s_single(allowed):
    ## This function takes in a single allowed index corresponding to a single segment, and spits out a list of allowed signature values at the relevant points. 
    sig_allowed = list(allowed+1)
    return sig_allowed

def allowed2s_double(allowed):
    ## This function takes in a single allowed index corresponding to a single segment, and spits out a list of allowed signature values at the relevant points. 
    ## First, we construct all allowable pairs. 
    allowed_pairs = [np.array([mi,mj])+1 for mi in allowed for mj in allowed if mi != mj or (mi==-1 and mj ==-1) ]
    ## We will construct a function to map these to integers as we would like: 
    to_double = lambda x: x[0]+x[1]*3
    sig_allowed = list(map(to_double,allowed_pairs))

    return sig_allowed

## The Configuration object that have built up can handle a lot of the machinery that we care about. However, the actual optimization of a query configuration should be handled by something that can easily manipulate the query object as necessary. handles two different estimation routines: 1. estimation by estimating the distance covered. 2. estimation by shotgun blackout of certain configuration entries. Give a traversal strategy: left to right, right to left, or both entries at once (should be okay if fast enough) 
## As a helper object, we introduce a ConstrainedNode class to allow us to traverse this easily. We have a depth argument, that indicates how far down the chain this node is. The maximum for this should be nb_segs-1
class ConstrainedNode(object):
    def __init__(self,signature,nb_segs,allowed_children,set_parent = None):
        assert signature is not None; 'signature should be a list'
        self.signature = signature
        self.length = nb_segs
        self.depth = len(signature)
        self.allowed_children = allowed_children
        ## This cost is a parameter that can only be set once! 
        self.cost = np.nan 
        if set_parent is None:
            self.parent = self.get_parent()
        else:
            self.parent = set_parent

    def children(self):
        child_sigs = self.childsig()
        return [ConstrainedNode(child_sig,self.length,self.allowed_children,set_parent = self) for child_sig in child_sigs]
    def childsig(self):
        if len(self.signature) == self.length:
            childsigs = None 
        else:
            childsigs = [self.signature+[a] for a in self.allowed_children[len(self.signature)]]
        return childsigs

    def set_cost(self,cost):
        assert np.isnan(self.cost) 
        self.cost = cost
    def get_cost(self):
        assert not np.isnan(self.cost) 
        return self.cost
        
    ## Returns the set of all nodes that describe the nth generation after this node. If the tree bottoms out, will return None. n = 1 returns children. 

    def descendants(self,n = 1):
        assert n>0
        ## First calculate if the depth is okay: 
        probe_depth = self.depth+n
        if probe_depth > self.length:
            generation = [None]
        else:
            ## Initialize holders: 
            new_parents = [self] 
            for ni in range(n):
                generation = []
                for parent in new_parents: 
                    generation = generation+parent.children()
                new_parents = generation
        return generation 
    def get_parent(self):
        parent_sig = self.parentsig()
        return ConstrainedNode(parent_sig,self.length,self.allowed_children)
    def parentsig(self):
        if len(self.signature) == 0:
            parentsig = None
        else:
            parentsig = self.signature[:-1]
        return parentsig

class Optimizer(object): 
    def __init__(self,conf,strategy,sort = 'LIFO'):
        self.conf = conf
        self.strategy = strategy 
        ## The strategy defines the conversion function to use and the overall length. 
        if strategy == 'left':
            self.s2q = lambda s: s2q_left(s,conf.nb_segs)
            self.s2q_default = lambda s,default: s2q_left_default(s,conf.nb_segs,default)
            self.length = conf.nb_segs*2
        elif strategy == 'right':
            self.s2q = lambda s: s2q_right(s,conf.nb_segs)
            self.s2q_default = lambda s,default: s2q_right_default(s,conf.nb_segs,default)
            self.length = conf.nb_segs*2
        elif strategy == 'both':
            self.s2q = lambda s: s2q_both(s,conf.nb_segs)
            self.s2q_default = lambda s,default: s2q_both_default(s,conf.nb_segs,default)
            self.length = conf.nb_segs
        self.construct_graph()
        if strategy is 'both':
            self.root_node = ConstrainedNode([],self.length,self.all_allowed,set_parent = [])
        else:
            allowed_rep = [a for a in self.all_allowed for i in range(2)]
            self.root_node = ConstrainedNode([],self.length,allowed_rep,set_parent = [])

        self.estopt = self.conf.estimate_optimal_post()
        self.sort = sort
        if sort == 'LIFO':
            self.current_nodes = [self.root_node]
        elif sort == 'Priority':
            self.current_nodes = []
            heappush(self.current_nodes,(np.sum(self.estopt),0,0,self.root_node))
        self.root_node.set_cost(0)
            
        self.boundcost = self.conf.full_cost()
        self.solutions = []
            
    ## Define the method to construct the allowable graph: 
    def construct_graph(self):
        ## Get all of the allowed indices first: 
        allowed = [self.conf.segs[i].allowed for i in range(self.conf.nb_segs)]
        allowed_full = [np.concatenate([a,[-1]]) for a in allowed]
        ## The conversion here depends upon the traversal strategy we will use: 
        if self.strategy == 'left' or self.strategy == 'right':
            func = allowed2s_single 
        elif self.strategy == 'both':
            func = allowed2s_double
        all_allowed = list(map(func,allowed_full))
        self.all_allowed = all_allowed

    ## BRANCH AND BOUNDDDDD
    ## Branch takes in a query signature, and generates child signatures from it unless there are no children and we are at the end of the line. 
    def branch(self,node):
        childsigs = node.childsig()
        if childsigs is None:
            relevant = None
        else:
            relevant = node.children()
        return relevant 

    ## A basic bounding strategy. Enumerate down to a depth of depth, and take the min over the estimated costs generated at that depth. 
    #def bound0(self,node,depth = 2):
    #    ## First clip the depth if necessary. 
    #    ## Take max of this and the true depth. 
    #    traverse_depth = np.min([depth,self.length-node.depth])
    #    if traverse_depth == 0:
    #        query = self.s2q(node.signature) 
    #        cost = self.conf.full_cost(query = query,check = False)
    #    else:
    #        descendants = node.descendants(traverse_depth)
    #        childsig = [des.signature for des in descendants]
    #        ## Now calculate estimated costs for all children signatures: 
    #        ## First convert children signatures to queries: 
    #        queries = list(map(self.s2q,childsig))
    #        ## Now estimate costs on the queries: 
    #        estimator = lambda queryc: self.conf.full_cost(query=queryc,check = False)
    #        costs = list(map(estimator,queries))
    #        cost = np.min(costs)
    #    return cost 
    
    ## Blackout a given query up to a certain point: 
    def blackout_query(self,q,node):
        query = np.copy(q) 
        if self.strategy == 'both':
            elim = node.depth 
            query[elim:] = -2
        elif self.strategy == 'left':
            elim = np.ceil(node.depth/2).astype(int)
            query[elim:] = -2
            if node.depth%2 == 1:
                query[elim-1,-1] = -2
        elif self.strategy == 'right':
            elim = np.ceil(node.depth/2).astype(int)
            query[elim:] = -2
            if node.depth%2 == 1:
                query[elim-1,0] =-2
        return query

    ## A check function to make sure that this generates desired answers.
    def bound0_new_check(self,node,depth = 2):
        ## Split up the trajectory into three parts: before the branch, after the branch, and during the branch: 
        traverse_depth = np.min([depth,self.length-node.depth])
        query = self.blackout_query(self.conf.work_config,node) 
        if traverse_depth == 0:
            cost = self.conf.full_cost(query = query,check = True)
        else:
            ## First calculate the cost on the query up to the current point. 
            if self.strategy == 'both':
                prolif_start = node.depth
                prolif_end = (node.depth+depth)
            else: 
                prolif_start = node.depth//2
                prolif_end = np.ceil((node.depth+depth+1)/2).astype(int) 
            precost = self.conf.cut_cost(query = query[:prolif_start],end = prolif_start,check = True)
            ## Now calculate the cost for after the branching is done too: 
            postcost = self.conf.cut_cost(query = query[prolif_end:],start = prolif_end,check = True)
            childqs = [self.blackout_query(self.conf.work_config,node.descendants(depth)[0])]
            childqs_trunc = [childqs[0][prolif_start:prolif_end]]
            ## Get the main costs in these parts: 
            branchcost = np.array([self.conf.cut_cost(query = q,start = prolif_start,end=prolif_end,check = True) for q in childqs_trunc]) 
            ## Get the bridge before and after: 
            bridgepre = np.array([self.conf.bridge_cost(query = q, split = prolif_start) for q in childqs]) 
            bridgepost = np.array([self.conf.bridge_cost(query = q, split = prolif_end) for q in childqs]) 
            ## Now calculate estimated costs for all children signatures: 
            return branchcost+bridgepre+bridgepost+precost+postcost


    def bound0_new(self,node,depth = 2):
        ## Split up the trajectory into three parts: before the branch, after the branch, and during the branch: 
        traverse_depth = np.min([depth,self.length-node.depth])
        query = self.s2q(node.signature) 
        if traverse_depth == 0:
            cost = self.conf.full_cost(query = query,check = False)
        else:
            ## First calculate the cost on the query up to the current point. 
            if self.strategy == 'both':
                prolif_start = node.depth
                prolif_end = (node.depth+depth)
            else: 
                prolif_start = node.depth//2
                prolif_end = np.ceil((node.depth+depth+1)/2).astype(int) 
            precost = self.conf.cut_cost(query = query[:prolif_start],end = prolif_start,check = False)
            ## Now calculate the cost for after the branching is done too: 
            postcost = self.conf.cut_cost(query = query[prolif_end:],start = prolif_end,check = False)
            ## Now the part that actually branches: 
            descendants = node.descendants(traverse_depth)
            ## Get the parts of the signature that actually differ: 
            childsigs = [des.signature for des in descendants]
            childqs = [self.s2q(sig) for sig in childsigs]
            childqs_trunc = [childq[prolif_start:prolif_end] for childq in childqs]
            ## Get the main costs in these parts: 
            branchcost = np.array([self.conf.cut_cost(query = q,start = prolif_start,end=prolif_end,check = False) for q in childqs_trunc]) 
            ## Get the bridge before and after: 
            bridgepre = np.array([self.conf.bridge_cost(query = q, split = prolif_start) for q in childqs]) 
            bridgepost = np.array([self.conf.bridge_cost(query = q, split = prolif_end) for q in childqs]) 
            ## Now calculate estimated costs for all children signatures: 
            ## First convert children signatures to queries: 
            cost = np.min(branchcost+bridgepre+bridgepost+precost+postcost)
        return cost

    ## This version cuts out the branching within the bound altogether. 
    def boundopt0(self,node,depth = 1):
        ## Split up the trajectory into two parts: before the branch, and during the branch: 
        query = self.s2q_default(node.signature,self.conf.work_config) 
        ## First calculate the cost on the query up to the current point. 
        if self.strategy == 'both':
            prolif_start = node.depth
        else: 
            prolif_start = node.depth//2
        parentcost = node.parent.get_cost()
        diffbridge_pre = self.conf.bridge_cost_opt(query = query,split=prolif_start-1,check = False)
        diffcost = self.conf.cut_cost(query = query[prolif_start-1:prolif_start],start = prolif_start-1,end = prolif_start,check = False)
        nodecost = parentcost+diffcost+diffbridge_pre
        node.set_cost(nodecost)
        ## Now calculate the cost for after the branching is done too: 
        postcost = np.nansum(self.estopt[prolif_start:])
        ## Get the bridge: 
        bridge = self.conf.bridge_cost(query = query, split = prolif_start) 
        cost = np.min(bridge+nodecost+postcost)
        return cost

    ## This version learns from boundopt0, and the fact that the heuristics are too low of a bound.  
    def boundopt1(self,node,depth = 5):
        ## Now split the trajectory into three parts: before the node,  
        ## Split up the trajectory into three parts: before the branch, after the branch, and during the branch: 
        traverse_depth = np.min([depth,self.length-node.depth])
        query = self.s2q_default(node.signature,self.conf.work_config) 
        if traverse_depth == 0:
            cost = self.conf.full_cost(query = query,check = False)
        else:
            ## First calculate the cost on the query up to the current point. 
            if self.strategy == 'both':
                prolif_start = node.depth
                prolif_end = (node.depth+traverse_depth)
            else: 
                prolif_start = node.depth//2
                prolif_end = np.ceil((node.depth+traverse_depth+1)/2).astype(int) 
            parentcost = node.parent.get_cost()
            diffcost = self.conf.cut_cost(query = query[prolif_start-1:prolif_start],start = prolif_start-1,end = prolif_start,check = False)
            diffbridge_pre = self.conf.bridge_cost_opt(query = query,split=prolif_start-1,check = False)
            nodecost = parentcost+diffcost+diffbridge_pre
            node.set_cost(nodecost)
            ## Now calculate the cost optimally: 
            optcost = np.nansum(self.estopt[prolif_start:prolif_end])
            ## Now calculate the cost after: 
            postcost = self.conf.cut_cost(query = self.boundquery[prolif_end:],start = prolif_end,check = False)
            ## Bridge optimally: 
            diffbridge_post = self.conf.bridge_cost_opt(query = self.boundquery,split =prolif_end,check =False)
            cost = nodecost+optcost+diffbridge_post+postcost 
        return cost




    def boundopt(self,node,depth = 1):
        ## Split up the trajectory into three parts: before the branch, after the branch, and during the branch: 
        traverse_depth = np.min([depth,self.length-node.depth])
        query = self.s2q_default(node.signature,self.conf.work_config) 
        if traverse_depth == 0:
            cost = self.conf.full_cost(query = query,check = False)
        else:
            ## First calculate the cost on the query up to the current point. 
            if self.strategy == 'both':
                prolif_start = node.depth
                prolif_end = (node.depth+traverse_depth)
            else: 
                prolif_start = node.depth//2
                prolif_end = np.ceil((node.depth+traverse_depth+1)/2).astype(int) 
            parentcost = node.parent.get_cost()
            diffcost = self.conf.cut_cost(query = query[prolif_start-1:prolif_start],start = prolif_start-1,end = prolif_start,check = False)
            diffbridge = self.conf.bridge_cost_opt(query = query,split=prolif_start-1,check = False)
            
            nodecost = parentcost+diffcost
            node.set_cost(nodecost)
            ## Now calculate the cost for after the branching is done too: 
            postcost = np.nansum(self.estopt[prolif_end:])
            ## Now the part that actually branches: 
            descendants = node.descendants(traverse_depth)
            ## Get the parts of the signature that actually differ: 
            childsigs = [des.signature for des in descendants]
            childqs = [self.s2q_default(sig,self.conf.work_config) for sig in childsigs]
            childqs_trunc = [childq[prolif_start:prolif_end] for childq in childqs]
            ## Get the main costs in these parts: 
            branchcost = np.array([self.conf.cut_cost(query = q,start = prolif_start,end=prolif_end,check = False) for q in childqs_trunc]) 
            ## Get the bridge before and after: 
            bridge = [self.conf.bridge_adaptive(query= q,prolif_start =prolif_start,prolif_end= prolif_end,check=False) for q in childqs]
            #bridgepre = np.array([self.conf.bridge_cost(query = q, split = prolif_start) for q in childqs]) 
            ### Should be beginning of estimation: use the optimal.
            #bridgepost = np.array([self.conf.bridge_cost_opt(query = q, split = prolif_end) for q in childqs]) 
            ## Now calculate estimated costs for all children signatures: 
            ## First convert children signatures to queries: 
            cost = np.min(branchcost+bridge+nodecost+postcost)
        return cost

    def bound(self,node,depth = 2):
        i = 0
        ## First clip the depth if necessary. 
        exploredepth = node.depth+depth
        ## Take max of this and the true depth. 
        traverse_depth = np.min([depth,self.length-node.depth])
        descendants = node.descendants(traverse_depth)
        childsig = [des.signature for des in descendants]
        ## Now calculate estimated costs for all children signatures: 
        ## First convert children signatures to queries: 
        queries = list(map(self.s2q,childsig))
        ## Now estimate costs on the queries: 
        split = len(node.signature)+depth
        estimator = lambda query: self.conf.estimate_cost(100,1,query=query,split = split,check = False)
        costs = list(map(estimator,queries))
        return np.min(costs)

    ### Now define a single branch-and-bound step: 
    #def step0(self,depth = 1):
    #    ## Get an element from the current node list: 
    #    node = self.current_nodes.pop()
    #    candidates = self.branch(node)
    #    if candidates is None: 
    #        query = self.s2q(node.signature)
    #        cost = self.conf.full_cost(query,check = False)
    #        if cost<self.boundcost:
    #            self.boundcost = cost
    #            self.solutions.append(node)
    #    else:
    #        for node in candidates:
    #            if self.bound0(node,depth = depth)>self.boundcost:
    #                pass
    #            else:
    #                self.current_nodes.append(node)
                    
    ## Now define a single branch-and-bound step: 
    def step0_new(self,depth = 1):
        ## Get an element from the current node list: 
        node = self.current_nodes.pop()
        candidates = self.branch(node)
        if candidates is None: 
            query = self.s2q(node.signature)
            cost = self.conf.full_cost(query,check = False)
            if cost<=self.boundcost:
                self.boundcost = cost
                self.solutions.append(node)
        else:
            for node in candidates:
                if self.bound0_new(node,depth = depth)>self.boundcost:
                    pass
                else:
                    self.current_nodes.append(node)

    def stepopt1(self,i,debug = False):
        ## Get an element from the current node list: 
        if self.sort == 'LIFO':
            node = self.current_nodes.pop()
        elif self.sort == 'Priority':
            cost,_,_,node = heappop(self.current_nodes)
        candidates = self.branch(node)
        if candidates is None: 
            print('solution')

            query = self.s2q(node.signature)
            cost = self.conf.full_cost(query,check = False)
            if cost<=self.boundcost:
                self.boundcost = cost
                self.solutions.append(node)
                ## Also have a query that corresponds to the current best solution
                self.boundquery = node.query
        else:
            for n,node in enumerate(candidates):
                l = len(candidates[0].signature)
                print(l)
                nodebound = self.boundopt1(node)
                if nodebound >self.boundcost:
                    pass
                    #print('pruned')
                else:
                    #print('passed')
                    if self.sort == 'LIFO':
                        self.current_nodes.append(node)
                    elif self.sort == 'Priority':
                        heappush(self.current_nodes,(nodebound,i+1,n,node))
                    if debug == True:
                        query = self.s2q(node.signature)
                        import pdb; pdb.set_trace()

    def stepopt0(self,i,debug = False):
        ## Get an element from the current node list: 
        if self.sort == 'LIFO':
            node = self.current_nodes.pop()
        elif self.sort == 'Priority':
            cost,_,_,node = heappop(self.current_nodes)
        candidates = self.branch(node)
        if candidates is None: 

            query = self.s2q(node.signature)
            cost = self.conf.full_cost(query,check = False)
            if cost<=self.boundcost:
                self.boundcost = cost
                self.solutions.append(node)
        else:
            for n,node in enumerate(candidates):
                l = len(candidates[0].signature)
                nodebound = self.boundopt0(node)
                if nodebound >self.boundcost:
                    pass
                else:
                    if self.sort == 'LIFO':
                        self.current_nodes.append(node)
                    elif self.sort == 'Priority':
                        heappush(self.current_nodes,(nodebound,i+1,n,node))
                    if debug == True:
                        query = self.s2q(node.signature)
                        import pdb; pdb.set_trace()


    def stepopt(self,depth = 1):
        ## Get an element from the current node list: 
        #cost,_,_,node = heappop(self.current_nodes)
        node = self.current_nodes.pop()
        print(len(node.signature))
        candidates = self.branch(node)
        if candidates is None: 

            query = self.s2q(node.signature)
            cost = self.conf.full_cost(query,check = False)
            if cost<=self.boundcost:
                self.boundcost = cost
                self.solutions.append(node)
        else:
            for n,node in enumerate(candidates):
                l = len(candidates[0].signature)
                nodebound = self.boundopt(node,depth = depth)
                print(nodebound,'bound')
                if nodebound >self.boundcost:
                    pass
                else:
                    #heappush(self.current_nodes,(nodebound,i,n,node))
                    self.current_nodes.append(node)


    ## Optimize:
    def optimize0(self,depth = 1):
        i = 0
        while len(self.current_nodes)>0:
            self.step0(depth)
            print(i)
            i+=1
    
    def optimize_opt1(self,debug = False):
        i = 0
        self.boundquery = self.conf.work_config
        while len(self.current_nodes)>0:
            self.stepopt1(i,debug = debug)
            i+=1

    def optimize_opt0(self,debug = False):
        i = 0
        while len(self.current_nodes)>0:
            self.stepopt0(i,debug = debug)
            i+=1

    def optimize_opt(self,depth = 1):
        i = 0
        while len(self.current_nodes)>0:
            self.stepopt(depth)
            i+=1

    def optimize(self,depth = 1):
        i = 0
        while len(self.current_nodes)>0:
            self.step(depth)
            print(i)
            i+=1

    ## Functions to handle solutions
    def solution_query(self,best = True):
        if len(self.solutions) < 0:
            print('no solutions found')
            costs = [None]
        else:
            if best == True:
                signature = self.solutions[-1].signature
                query = self.s2q(signature)
            else:
                signatures = [node.signature for node in self.solutions]
                query = list(map(self.s2q,signatures))
        return query

    ## Functions to handle solutions
    def solution_costs(self,best = True):
        if len(self.solutions) < 0:
            print('no solutions found')
            costs = [None]
        else:
            if best == True:
                signature = self.solutions[-1].signature
                query = self.s2q(signature)
                cost = self.conf.full_cost(query)
            else:
                signatures = [node.signature for node in self.solutions]
                queries = list(map(self.s2q,signatures))
                cost = list(map(self.conf.full_cost,queries))
        return cost
    
    def groundtruth_cost(self,test):
        return self.conf.full_cost(generate_groundtruth(test))

    def solution_plots(self,best = True):
        if len(self.solutions) < 0:
            print('no solutions found')
            costs = [None]
        else:
            if best == True:
                signature = self.solutions[-1].signature
                query = self.s2q(signature)
                self.conf.plot_full_cost(query)
            else:
                signatures = [node.signature for node in self.solutions]
                queries = list(map(self.s2q,signatures))
                list(map(self.conf.plot_full_cost,queries))

    def time_step(self,step):
        start = timer()
        step(0) 
        end = timer()
        print(end-start)


    ## Check the cost at the optimal configuration, and all others (allowed and unallowed). Notably, check that the optimum is locally stable. 
## Write a function to generate the groundtruth configuration given a test object. 
def generate_groundtruth(test): 
    ## What's important when given a test object is the flips. 
    flips = test.flips
    ## These flips follow pull logic: the position of a zero takes the value of the other trajectory. 
    ## The correct thing to do to correct these is as follows: 
    # If the other trajectory is given, we should delete it. If it is not, we should flip them. Otherwise, we should leave it as is.  
    ## Initialize the array: 
    gtquery = np.ones(np.shape(flips))*-2
    ## Create an answer key flip array: 
    fliparray = np.zeros(np.shape(flips))
    fliparray[:,0] +=1 
    ## Find all flipped positions: 
    for f,flipentry in enumerate(flips): 
        if sum(flipentry) == 2:
            pass
        elif sum(flipentry) == 1:
            mouse = np.where(flipentry == 0)[0][0]
            gtquery[f,mouse] = -1
        elif sum(flipentry) == 0:
            gtquery[f,:] = fliparray[f,:]

    ## Now tell gtquery what to do at these points:
    #gtquery[] = fliparray[]
    return gtquery.astype(int)

def check_optimum(config,test):
    ## Find the groundtruth (unswitched) configuration: 
    gtquery = generate_groundtruth(test) 
    ## Also get the current base configuration: 
    basequery = config.base_config
    
    ## We will perform two checks. 1. Check that the optimum cost with optimal lagrange is higher for the ground truth than for all local perturbations. 2. Check that correcting the base configuration towards the ground truth leads to an improving cost. 

    ## 1: 
    ## First creat a dummy query configuration: 
    dummyquery = np.zeros(basequery.shape)
    dummyquery[:,:] = basequery
    dummyquery[np.where(gtquery!=-2)] = gtquery[np.where(gtquery!=-2)]
    ## Record the cost: 
    gt_cost = config.full_cost(dummyquery.astype(int))
    ## Initialize the difference array: 
    diff1 = np.zeros((dummyquery.shape[0],dummyquery.shape[1],3))
    ## Now iterate through changing one entry at a time: 
    for j in range(len(dummyquery)):
        for i in range(2):
            ## Check if this is allowed: 
            if dummyquery[j,abs(1-i)] != -1:
                ## If allowed, create an array of allowed values here: 
                allowed = np.concatenate((config.segs[j].allowed,[-1])) 
                for k,kval in enumerate(allowed): 
                    pertquery = np.zeros(dummyquery.shape)
                    pertquery[:,:] = dummyquery
                    pertquery[j,i] = kval 
                    pert_cost = config.full_cost(pertquery.astype(int))
                    diff1[j,i,k] = pert_cost-gt_cost
    ## First creat a dummy query configuration: 
    dummyquery = np.zeros(basequery.shape)
    dummyquery[:,:] = basequery
    ## Record the cost: 
    ## Find the length of the path from base to ground truth: 
    wa,wb = np.where(gtquery!=-2)
    iterinds = zip(wa,wb)
    ## Initialize the difference array: 
    diff2 = np.zeros(len(wa))
    ## Now iterate through changing one entry at a time: 
    for j,jinds in enumerate(iterinds):
        dummyquery[jinds] = gtquery[jinds] 
        pert_cost = config.full_cost(dummyquery.astype(int))
        diff2[j] = pert_cost-gt_cost
    return diff1,diff2

## Lay out an expansion around the ground truth at every point. 
def linearize_path(opt,test):
    ## Get the root node: 
    node = opt.root_node
    ## Get the groundtruth query: 
    gt_query = opt.conf.return_checked(generate_groundtruth(test))
    ## Convert this to a signature: 
    bump = gt_query+1
    sig = bump[:,0]+bump[:,1]*3
    ## Now iterate through the whole segment set once: 
    all_tings = []
    for i in range(opt.conf.nb_segs):
        children = node.children()
        childqs = [c.signature for c in children]
        childindex = np.where(childqs == sig[i])[0][0]
        node = children[childindex]
        boundchild = np.array([opt.boundopt1(cn,depth = 4) for cn in children])
        boundchild = boundchild-boundchild[childindex]
        all_tings.append(boundchild)
    return np.array(all_tings)

        
        
    

    



    ## Use the optimal regularization terms to calculate the best way to estimate the cost.
    ## Estimate the cost. This happens by looking at inter distances, and only locally to the point of interest. 1. Calculate a per-segment tortuosity rate, and aggregate into an average. Then, estimate the distance traveled over the rest of the trajectory by sampling points, and applying that rate to the distance traveled. Check: The ranking of configurations near the optimum for ground truth vs. estimation. 
    ## Plotting functions



    
