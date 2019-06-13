import numpy as np
## A python module that holds the refactored code for behavioral trace analysis. This new code is based on the fundamental unit of a segment object, that is organized and held in a configuration object. This module should be independent of all the modules that have been written previously, but is testable using the SwitchCircle object in Social_Dataset_BB_testdata. 

## The fundamental unit of our new analysis. Contains segments of behavioral traces, chunked in time such that handling by higher order functions is simple and efficient. Carries a base, working, and query configuration to handle flexible working with multiple hypothetical  

invalid_int = -999999
## Helper function
## Handles indexing into numpy arrays using configuration arrays. 
def c2i(data,config):
    out = data[config]
    out[np.where(config == -1)] = invalid_int
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
    conf = mask*np.array([[1,2]])
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
        self.trajs = trajectories 
        ## We have trajectory a and b: this is the way the trajectories are initialized. These attributes should be 100% immutable. 
        self.traja = trajectories[:,0:2]
        self.trajb = trajectories[:,2:4]
        ## For indexing purposes: 
        self.timeindex = timeindex
        self.length = self.timeindex[-1]-self.timeindex[0]
        self.segindex = segindex
        ## Check for consistency between the attributes we have declared thus far.  
        assert len(self.trajs) == self.timeindex[1]-self.timeindex[0],'timeindex and trajectories must match '
        ## Finally, initialize the configuration of the two trajectories: 
        self.base_config = init_config
        self.barred = np.where(self.base_config==-1)[0]
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
    def get_dist(self,mouse = None, query = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize distances according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                dists = c2i(self.dists,query) 
                
            else: 
                dists = np.array([np.nan,np.nan])
        else: 
            dists = c2i(self.dists,self.work_config) 
        ## Now index into the distances 
        if mouse is None: 
            dist= np.sum(dists)
        elif mouse in [0,1]:
            dist = dists[mouse]
        else:
            raise ValueError
        return dist
            
    def get_tv(self,mouse = None, query = None):
        '''
        Function to handle configuration information and return parameters as appropriate.Can specify a query configuration, and a mouse. If not, sum across both mice for working configuration will be returned.  
        '''
        ## Organize tv according to the configuration. 
        if query is not None: 
            if self.check_config(query):
                tvs = c2i(self.tvs,query) 
                
            else: 
                tvs = np.array([np.nan,np.nan])
        else: 
            tvs = c2i(self.tvs,self.work_config) 
        ## Now index into the tvances 
        if mouse is None: 
            tv= np.sum(tvs)
        elif mouse in [0,1]:
            tv = tvs[mouse]
        else:
            raise ValueError
        return tv
            
            
    def get_start(self,mouse = None, query = None):
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

    def get_end(self,mouse = None, query = None):
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

    def plot_trace():
        pass

## We define a configuration class that holds a list of segment objects, and can test various manipulations of them. As with individual segments, the configuration object has different "tiers" of configuration, starting with the base configuration (no touch) to query configuration that are just passed to the data for the sake of testing out a particular configuration. The configuration code for Configurations has one more flag compared to the Segment: -2 says, we are agnostic, reference the current underlying trajectory. 
class Configuration(object):
    ## Lagrange multipliers are for the first and second penalty term when calculating the cost. 
    def __init__(self,segs,lagrange1,lagrange2):
        self.segs = segs
        self.weigths = [lagrange1,lagrange2]
        self.nb_segs = len(self.segs)
        self.length = np.sum([seg.length for seg in self.segs])
        self.base_config = np.stack([seg.base_config for seg in self.segs],axis = 0)
        self.set_work_config(self.base_config)

    ## We have one more 
    def check_config(self,config):
        assert len(config) == self.nb_segs
        ## Segmentwise check. First take the segment list, the relevant configs, and zip: 
        inputs = zip(self.segs,config)
        ## Now broadcast with map. 
        checks = list(map(lambda x: Segment.check_config(*x),inputs))
        ## The configuration is only valid if all the entires are individually valid. 
        check = np.all(checks)
        return check

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

    ## Now, some functions to allow for easy indexing. Let's see if we can pull out the trajectory of a 

    
        






            

        







        
        



